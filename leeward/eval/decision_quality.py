"""Decision quality -- the Impact bar chart. SPEC §11.

For each held-out day and each call budget K, four strategies pick who the care team calls:

    leeward          leeward.decision.allocate under a K-call budget
    rank_by_age      the K oldest veterans, a care_team_call each
    rank_by_chronic  the K veterans with the most chronic conditions, a care_team_call each
    random           K veterans from a shuffle seeded at 0 (and keyed by the day)

and each is scored against what actually happened in the simulation:

    harm averted = Σ w_k · τ[a,k] · y_true[i,k,t]    over the veterans selected

Every strategy gets the same budget and nothing else: K call-bucket slots, zero rides,
refills, bookings or free texts. The baselines are what a team does today when it works a
list top-down; Leeward chooses both who and which call. If one veteran receives several
actions, their effect on one need combines as 1 - Π(1 - τ) -- identical to the formula for a
single action, and it cannot claim more than all of an event was prevented.

`w_k` and `τ[a,k]` come from the decision layer, so the allocator is scored on the same
numbers it optimises. That layer (`leeward.decision`, api lane) is the only dependency this
module has on unbuilt code, and it is imported lazily: everything else here runs today.

    python -m leeward.eval.decision_quality     # -> report/decision_quality.{csv,html}
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from leeward import schema
from leeward.schema import ACTION_COST_UNIT, DEFAULT_CAPACITY, NEEDS

REPORT = schema.ROOT / "report"

KS = (20, 40, 80)
HOLDOUT_DAYS = 30                 # days 91-120 of the 120-day simulation
RANDOM_SEED = 0
BASELINE_ACTION = "care_team_call"
BUDGET_BUCKET = ACTION_COST_UNIT[BASELINE_ACTION]

#: Order matters: it is the column order of the chart and of `summarise`.
STRATEGIES = ("leeward", "rank_by_age", "rank_by_chronic", "random")
BASELINES = STRATEGIES[1:]

TIDY_SCHEMA = {
    "date": pl.Date, "k": pl.Int32, "strategy": pl.Utf8,
    "harm_averted": pl.Float64, "n_veterans": pl.Int32, "n_actions": pl.Int32,
}

Weights = Mapping[str, float]
Tau = Mapping[str, Mapping[str, float]]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

def decision_weights() -> tuple[Weights, Tau]:
    """`w_k` and `τ[a,k]` from the decision layer: the numbers the allocator optimises.

    Both are YAML-backed so a clinician can retune them without touching code, which is
    what `docs/SPEC.md` 7.1 and 7.2 ask for -- so read them through `load()` rather than
    importing a module constant. Actions with no τ row, like the information-only
    check_in_call, avert no measured harm.
    """
    from leeward.decision import severity, tau
    return severity.load(), tau.load()


def holdout_dates(scores: pl.DataFrame, outcomes: pl.DataFrame,
                  n_days: int = HOLDOUT_DAYS) -> list[date]:
    """The last `n_days` dates that have both scores and outcomes: days 91-120 of a
    120-day run; every day of a shorter fixture."""
    both = set(scores["date"].unique().to_list()) & set(outcomes["date"].unique().to_list())
    if not both:
        raise ValueError("scores and outcomes share no dates; nothing to evaluate")
    return sorted(both)[-n_days:]


def call_budget(k: int) -> dict[str, int]:
    """K calls and nothing else -- the budget every strategy gets."""
    return {bucket: 0 for bucket in DEFAULT_CAPACITY} | {BUDGET_BUCKET: k}


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

def baseline_actions(strategy: str, panel: pl.DataFrame, day: date, k: int,
                     seed: int = RANDOM_SEED) -> pl.DataFrame:
    """Top-K of today's panel by a rule that ignores the model, one generic call each.

    Ties fall to `veteran_id`, and random shuffles a panel sorted by id, so the selection
    never depends on the order the cohort arrived in.
    """
    ordered = panel.sort("veteran_id")
    if strategy == "rank_by_age":
        ids = ordered.sort("age", descending=True, maintain_order=True)["veteran_id"]
    elif strategy == "rank_by_chronic":
        ids = ordered.sort("n_chronic", descending=True, maintain_order=True)["veteran_id"]
    elif strategy == "random":
        rng = np.random.default_rng([seed, day.toordinal()])
        ids = ordered["veteran_id"].gather(rng.permutation(ordered.height))
    else:
        raise ValueError(f"unknown baseline strategy {strategy!r}; expected one of {BASELINES}")
    ids = ids.head(k)
    return pl.DataFrame({
        "veteran_id": ids,
        "date": pl.Series([day] * ids.len(), dtype=pl.Date),
        "action": [BASELINE_ACTION] * ids.len(),
    })


def leeward_actions(allocate: Callable[..., pl.DataFrame], scores_day: pl.DataFrame,
                    cohort: pl.DataFrame, k: int) -> pl.DataFrame:
    """Leeward's pick under the same K-call budget, checked so the comparison stays fair."""
    actions = allocate(scores=scores_day, cohort=cohort, capacity=call_budget(k))
    schema.validate(actions, "actions")
    outside = actions.filter(pl.col("capacity_bucket") != BUDGET_BUCKET)
    if outside.height or actions.height > k:
        raise ValueError(
            f"allocate() was given {k} calls and nothing else but returned {actions.height} "
            f"actions ({outside.height} outside the call bucket). The baselines get exactly K "
            "calls, so this would not be a like-for-like comparison.")
    return actions.select("veteran_id", "date", "action")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _check_weights(w: Weights, tau: Tau) -> None:
    missing = [k for k in NEEDS if k not in w]
    if missing:
        raise ValueError(f"severity weights w_k missing for needs {missing}")
    for action, row in tau.items():
        bad = {k: v for k, v in row.items() if not 0.0 <= v <= 1.0}
        if bad:
            raise ValueError(f"τ[{action}] must be a fraction in [0, 1]; got {bad}")


def harm_averted(actions: pl.DataFrame, outcomes: pl.DataFrame, w: Weights, tau: Tau) -> float:
    """Σ w_k · τ[a,k] · y_true[i,k,t] over the selected veteran-days.

    `actions` needs veteran_id, date, action; `outcomes` is the outcomes contract. Several
    actions on one veteran-need combine as 1 - Π(1 - τ).
    """
    _check_weights(w, tau)
    if actions.is_empty():
        return 0.0
    picked = actions.select("veteran_id", "date", "action")
    joined = picked.join(outcomes.select("veteran_id", "date", "need", "y"),
                         on=["veteran_id", "date"], how="left")
    ghosts = joined.filter(pl.col("need").is_null())
    if ghosts.height:
        raise ValueError(
            f"{ghosts.height} selected veteran-days have no outcomes, e.g. "
            f"{ghosts.select('veteran_id', 'date').row(0)}. Scoring them as 0 would quietly "
            "undercount whichever strategy picked them.")

    tau_rows = pl.DataFrame(
        [(a, k, float(v)) for a, row in tau.items() for k, v in row.items()],
        schema={"action": pl.Utf8, "need": pl.Utf8, "tau": pl.Float64}, orient="row")
    per_need = (joined.join(tau_rows, on=["action", "need"], how="left")
                      .with_columns(pl.col("tau").fill_null(0.0))
                      .group_by("veteran_id", "date", "need")
                      .agg((1 - (1 - pl.col("tau")).product()).alias("prevented"),
                           pl.col("y").first()))
    weight = pl.col("need").replace_strict(dict(w), return_dtype=pl.Float64)
    return float(per_need.select((weight * pl.col("prevented") * pl.col("y")).sum()).item())


def evaluate(scores: pl.DataFrame, cohort: pl.DataFrame, outcomes: pl.DataFrame, *,
             w: Weights, tau: Tau, ks: Sequence[int] = KS,
             dates: Sequence[date] | None = None, seed: int = RANDOM_SEED) -> pl.DataFrame:
    """Harm averted per day x K x strategy, tidy. Dates default to the held-out window."""
    from leeward.decision.allocate import allocate  # the one piece of the api lane we need

    _check_weights(w, tau)
    dates = holdout_dates(scores, outcomes) if dates is None else list(dates)
    rows = []
    for day in dates:
        scores_day = scores.filter(pl.col("date") == day)
        outcomes_day = outcomes.filter(pl.col("date") == day)
        panel = cohort.join(scores_day.select("veteran_id").unique(), on="veteran_id",
                            how="semi")
        for k in ks:
            picks = {"leeward": leeward_actions(allocate, scores_day, cohort, k)}
            picks |= {b: baseline_actions(b, panel, day, k, seed) for b in BASELINES}
            for strategy in STRATEGIES:
                acts = picks[strategy]
                rows.append({
                    "date": day, "k": k, "strategy": strategy,
                    "harm_averted": harm_averted(acts, outcomes_day, w, tau),
                    "n_veterans": acts["veteran_id"].n_unique(),
                    "n_actions": acts.height,
                })
    return pl.DataFrame(rows, schema=TIDY_SCHEMA)


def summarise(tidy: pl.DataFrame) -> pl.DataFrame:
    """Mean harm averted per day, one row per (K, strategy) -- `DecisionQualityRow`'s shape.

    Per day, because K is a daily budget: the same unit as the harm-averted counter beside
    the capacity slider.
    """
    order = pl.Enum(list(STRATEGIES))
    return (tidy.group_by("k", "strategy")
                .agg(pl.col("harm_averted").mean(), pl.len().cast(pl.Int32).alias("n_days"))
                .sort(pl.col("k"), pl.col("strategy").cast(order)))


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

LABELS = {"leeward": "Leeward", "rank_by_age": "Rank by age",
          "rank_by_chronic": "Rank by chronic-condition count", "random": "Random (seed 0)"}
# Categorical slots 1-4 in fixed order; validated for adjacent-pair CVD separation.
COLORS = {"leeward": "#2a78d6", "rank_by_age": "#eb6834",
          "rank_by_chronic": "#1baf7a", "random": "#eda100"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"


def bar_chart(tidy: pl.DataFrame) -> go.Figure:
    """Grouped columns: K on the x-axis, one column per strategy, mean harm averted per day."""
    summary = summarise(tidy)
    first, last = tidy["date"].min(), tidy["date"].max()
    fig = go.Figure()
    for strategy in STRATEGIES:
        s = summary.filter(pl.col("strategy") == strategy)
        fig.add_bar(
            name=LABELS[strategy],
            x=[f"{k} calls a day" for k in s["k"]],
            y=s["harm_averted"].to_list(),
            marker={"color": COLORS[strategy], "cornerradius": 4},
            # Label only the series the story is about; the rest are in the hover and the CSV.
            texttemplate="%{y:.1f}" if strategy == "leeward" else None,
            textposition="outside",
            textfont={"color": INK},
            hovertemplate=(f"<b>{LABELS[strategy]}</b><br>%{{x}}<br>"
                           "%{y:.2f} severity-weighted events averted per day<extra></extra>"),
        )
    fig.update_layout(
        title={"text": "Harm averted per day, by who the care team calls",
               "subtitle": {"text": (f"Held-out days {first} to {last} · every strategy gets "
                                     "the same K calls · simulated outcomes, synthetic cohort"),
                            "font": {"color": INK_2, "size": 13}},
               "font": {"color": INK, "size": 18}, "x": 0.02, "xanchor": "left"},
        barmode="group", bargap=0.5, bargroupgap=0.1,
        width=760, height=460, margin={"l": 64, "r": 24, "t": 96, "b": 56},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": INK_2},
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": -0.14,
                "font": {"color": INK_2}},
        xaxis={"showgrid": False, "linecolor": AXIS, "tickfont": {"color": MUTED}},
        yaxis={"title": {"text": "Severity-weighted events averted per day",
                         "font": {"color": INK_2, "size": 12}},
               "rangemode": "tozero", "gridcolor": GRID, "gridwidth": 1,
               "zerolinecolor": AXIS, "tickfont": {"color": MUTED}, "separatethousands": True},
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": INK}},
    )
    return fig


def write_outputs(tidy: pl.DataFrame, out_dir: Path = REPORT) -> tuple[Path, Path]:
    """report/decision_quality.csv (tidy, per day) and .html (Plotly, works offline)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, html = out_dir / "decision_quality.csv", out_dir / "decision_quality.html"
    tidy.select(list(TIDY_SCHEMA)).sort("date", "k").write_csv(csv)
    # include_plotlyjs=True inlines plotly.js: bigger file, but it opens with the wifi off.
    bar_chart(tidy).write_html(html, include_plotlyjs=True, full_html=True)
    return csv, html


def main() -> int:
    w, tau = decision_weights()
    tidy = evaluate(schema.read("scores"), schema.read("cohort"), schema.read("outcomes"),
                    w=w, tau=tau)
    csv, html = write_outputs(tidy)
    with pl.Config(tbl_rows=len(KS) * len(STRATEGIES)):
        print(summarise(tidy))
    print(f"wrote {csv.relative_to(schema.ROOT)} and {html.relative_to(schema.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

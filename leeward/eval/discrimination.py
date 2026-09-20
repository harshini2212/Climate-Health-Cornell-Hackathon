"""Discrimination -- can Leeward tell today's sick veteran from today's well one? SPEC §11.

[TRIPOD+AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/), the 2024 reporting standard
for clinical prediction models, names three things: discrimination, calibration and clinical
utility. `calibration.py` does the second, `decision_quality.py` the third, and until this
module existed nothing in the repo did the first -- which matters more here than the missing
leg usually does, because at these base rates calibration cannot separate Leeward from a
predictor that has never met anyone. A single number equal to the base rate scores ECE 0.0000
(`calibration.constant_ece`, and the reference markers on the reliability chart). The
difference between the model and that number is entirely discrimination.

**Within-day AUC is the headline; pooled AUC is the flattering one.** The care team gets a
call budget every morning and chooses a list from the people scored *that day*, so the only
ranking question anyone acts on is "of today's panel, who?". Pooled AUC answers a question
nobody asks -- it mixes in "was today a heat wave", which the model can get right by reading
the forecast and which buys the team nothing, because they cannot call Tuesday's panel on
Monday. Both numbers are reported side by side; the gap between them is the day effect.

This is not a coinage. The estimand is the **within-cluster concordance probability**, the
cluster being the day, and van Klaveren, Steyerberg, Perel & Vergouwe (BMC Med Res Methodol
2014;14:5) make exactly this argument: "the within-cluster concordance probability is most
relevant when a risk model supports decisions within clusters." TRIPOD-Cluster asks authors
to say which version of the c-statistic they computed, so, plainly: **this is the unweighted
mean of per-day c-statistics** -- one vote per day, because the team has a budget on the quiet
day too. That is a deliberate choice of estimand, not the only one; a usable-pairs-weighted
mean, or Janes & Pepe's covariate-adjusted ROC, answers "the typical within-day comparison"
instead of "the typical day", and would give a slightly different number. `daily_auc` returns
the whole distribution so the spread is inspectable rather than hidden behind the mean.

**A day on which nobody had the event has no ranking to score, and is dropped.** Folding those
in as 0.5 would drag every rare need toward a coin flip and present that as the model's
failure. But dropping them conditions on the outcome, which tilts the surviving days toward
the eventful ones -- at a 0.34% base rate that is a real selection effect and not a rounding
detail. So `n_days` and `n_days_total` are both reported, and the demo says the ratio out loud
rather than quoting a mean over an unstated subset.

Lift is the operator-facing number: select the top q% of *each day's* panel and the event rate
among them is this many times the base rate. It is improper and prevalence-dependent, so it is
reported with its ceiling -- lift@q can never exceed min(1/q, 1/base_rate) however good the
model is, and at a 2.33% base rate the top 1% is capped at 42.9x, not 100x.

PR-AUC (average precision, never trapezoidal interpolation, which is invalid in PR space --
Davis & Goadrich, ICML 2006) is reported as a **secondary descriptive** figure only. It is
tempting to promote it at these base rates, and the current guidance says not to: STRATOS TG6
(Van Calster et al., Lancet Digit Health 2025) "do not agree with researchers who recommend to
use AUPRC ... instead of AUROC", because it has no clear interpretation, depends on prevalence
and ignores true negatives; and McDermott et al. (NeurIPS 2024) show AUPRC "can unduly favor
model improvements in subpopulations with more frequent positive labels", which is a fairness
hazard in a project that ships a fairness audit. AUROC stays the headline.

No scikit-learn. AUC is the Mann-Whitney rank identity with tied scores given their average
rank -- a tie is half a win, because a score that cannot separate two people has not.

    python -m leeward.eval.discrimination     # -> report/discrimination.{csv,html}
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from leeward import schema
from leeward.eval.calibration import paired
from leeward.schema import NEEDS

REPORT = schema.ROOT / "report"

#: The two slices of the call list worth naming: a 40-call day on 10,000 veterans is 0.4%.
LIFT_QUANTILES = (0.01, 0.10)

DISCRIMINATION_SCHEMA = {
    "need": pl.Utf8,
    "within_day_auc": pl.Float64,
    "pooled_auc": pl.Float64,
    "pr_auc": pl.Float64,
    "scaled_brier": pl.Float64,
    "brier": pl.Float64,
    "lift_at_1pct": pl.Float64,
    "lift_at_10pct": pl.Float64,
    "lift_ceiling_1pct": pl.Float64,
    "base_rate": pl.Float64,
    "n": pl.Int32,
    "n_events": pl.Int32,
    "n_days": pl.Int32,
    "n_days_total": pl.Int32,
}

ArrayLike = Iterable[float] | pl.Series | np.ndarray


# --------------------------------------------------------------------------- #
# The rank identity
# --------------------------------------------------------------------------- #

def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their group's mean rank -- `scipy.stats.rankdata`.

    Written out because a tie-break would let the order rows arrived in move the headline
    number, and because scipy is not a dependency of this project.
    """
    order = np.argsort(x, kind="stable")
    ordered = x[order]
    _, first, inverse, counts = np.unique(ordered, return_index=True, return_inverse=True,
                                          return_counts=True)
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = (first + 1 + (counts - 1) / 2.0)[inverse]
    return ranks


def _clean(p: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(np.int64)
    if p.shape != y.shape:
        raise ValueError(f"scores and outcomes differ in length: {p.shape} vs {y.shape}")
    return p, y


def auc(p: ArrayLike, y: ArrayLike) -> float | None:
    """P(a random event scored above a random non-event), a tie counting half.

    `None` when every row is an event or none is: there are no pairs, so there is no
    ranking to be right or wrong about, and 0.0 would read as "every pair backwards".
    """
    p, y = _clean(p, y)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(p)
    u = float(ranks[y == 1].sum()) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def brier(p: ArrayLike, y: ArrayLike) -> float | None:
    """Mean squared error of the probabilities. Strictly proper, and prevalence-dependent.

    Reported only so that `scaled_brier` beside it cannot be mistaken for it. A raw Brier
    score looks wonderful at a 0.34% base rate -- predicting zero for everybody scores
    0.0034 -- which says nothing about the model and everything about how rare the event is.
    """
    p, y = _clean(p, y)
    return float(np.mean((p - y) ** 2)) if p.size else None


def scaled_brier(p: ArrayLike, y: ArrayLike) -> float | None:
    """1 - Brier / Brier_null: the Brier skill score, a.k.a. the Index of Prediction Accuracy.

    **This is the number that settles the argument `calibration.constant_ece` opens.** The
    null model is the constant at the base rate -- the same predictor that scores a perfect
    ECE -- and its Brier is exactly π(1-π), so it scores **0.0 here, by construction**, while
    a perfect predictor scores 1.0 and anything worse than the base rate goes negative.

    It works where ECE cannot because it is *strictly* proper: under the Murphy decomposition
    it keeps the sharpness term that a calibration-only measure discards, so it is uniquely
    minimised by the truth rather than by any well-calibrated blur of it. STRATOS TG6 (Van
    Calster et al., Lancet Digit Health 2025) recommends exactly this scaling, for exactly
    this reason: the raw Brier is dominated by prevalence, and the skill score divides that out.

    **What it says about rung 0, which is not a flattering answer and is the useful one.**
    On the real prior-only scores: heat +0.110, treatment_gap +0.042, breathing +0.001,
    mental -0.004, access_loss -0.014. So on a strictly proper score the three rarest needs
    are *at or below* a constant at the base rate, even though all three rank above chance
    (within-day AUC 0.58-0.65). Ranking and scale are different things: rung 0 has never seen
    an outcome, so it orders people better than a coin while its probabilities are still the
    priors' -- and at a 0.34% base rate an ECE of 0.0067 is twice the base rate itself, which
    costs more in squared error than the weak ranking earns back.

    Say this out loud rather than quoting heat alone. It is the clearest argument in the repo
    for rung 1: the ranking is already there, the calibration of the rare needs is what a fit
    would buy. And it is why AUROC stays the headline for the call-list claim -- the list is
    an ordering decision, and the ordering is real even where the scale is not.
    """
    p, y = _clean(p, y)
    base = float(np.mean(y)) if y.size else 0.0
    reference = base * (1.0 - base)          # the constant-at-base-rate Brier, in closed form
    if reference <= 0.0:
        return None                          # no events, or nothing but events: no skill to have
    return 1.0 - float(np.mean((p - y) ** 2)) / reference


def average_precision(p: ArrayLike, y: ArrayLike) -> float | None:
    """PR-AUC as average precision: Σ (recall_i - recall_{i-1}) · precision_i.

    Rows sharing a score are taken as one step. Splitting them would invent a ranking the
    model did not supply, and on a constant predictor that invention scores up to 1.0.
    """
    p, y = _clean(p, y)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-p, kind="stable")
    hits = y[order]
    # One step per distinct score: the last index of each run of equal scores.
    step = np.append(np.flatnonzero(np.diff(p[order]) != 0), p.size - 1)
    tp = np.cumsum(hits)[step]
    selected = step + 1
    precision = tp / selected
    recall = tp / n_pos
    return float((precision * np.diff(np.concatenate(([0.0], recall)))).sum())


# --------------------------------------------------------------------------- #
# Per need, over the held-out window
# --------------------------------------------------------------------------- #

def _one_need(pairs: pl.DataFrame) -> pl.DataFrame:
    """Guard against being handed several needs at once, which would pool their base rates."""
    needs = pairs["need"].unique().to_list()
    if len(needs) > 1:
        raise ValueError(f"expected one need, got {sorted(needs)}; filter first -- pooling "
                         "needs would average base rates that differ by a factor of seven")
    return pairs


def daily_auc(pairs: pl.DataFrame) -> list[float]:
    """That day's AUC, for every held-out day on which the need actually happened."""
    _one_need(pairs)
    out = []
    for (_,), part in pairs.group_by("date"):
        value = auc(part["p_mean"], part["y"])
        if value is not None:
            out.append(value)
    return out


def within_day_auc(pairs: pl.DataFrame) -> float | None:
    """The mean over days of that day's AUC -- the number that justifies a call list.

    Not the pooled AUC, and not a row-weighted mean of the daily ones: the care team has a
    budget on the quiet day too, so a day where the model ranked well counts the same whether
    three people or three hundred had an event.
    """
    values = daily_auc(pairs)
    return float(np.mean(values)) if values else None


def lift(pairs: pl.DataFrame, q: float) -> float | None:
    """Event rate in the top `q` of each day's panel, over the whole window's base rate.

    Selected *per day*, because that is when the list is made. Picking the window's top q%
    in one pass would spend the whole month's budget on the heat wave and call nobody on the
    other twenty-nine days -- a schedule no team can work. It is not a small difference and
    it flatters: on the real rung-0 scores, heat's lift@1% is **10.7x selected per day and
    18.2x pooled**, and the gap is entirely the day effect, exactly as it is for AUC. The
    two needs with no day effect (breathing, mental) score the same either way.

    Ties fall to `veteran_id`, so the same click gives the same number.
    """
    _one_need(pairs)
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be a fraction in (0, 1]; got {q}")
    base = pairs["y"].mean()
    if not base:
        return None
    # Each day gets its own top q%, sized from its own panel: a day on which half the cohort
    # was not scored still gets a list, it is just a shorter one.
    place = pl.int_range(pl.len()).over("date")
    budget = (pl.len().over("date") * q).ceil().clip(lower_bound=1)
    picked = (pairs.sort(["p_mean", "veteran_id"], descending=[True, False])
                   .filter(place < budget))
    return float(picked["y"].mean() / base)


def lift_ceiling(pairs: pl.DataFrame, q: float) -> float | None:
    """The best lift@q any predictor could score here: min(1/q, 1/base_rate).

    Reported next to the lift because without it the number is uninterpretable. Calling the
    top 1% cannot beat 42.9x on a need that happens to 2.33% of people, however perfect the
    ranking -- there are not enough events to fill the list. A reader who does not know the
    ceiling cannot tell 10.7x of a possible 42.9x from 10.7x of a possible 11.
    """
    base = pairs["y"].mean()
    return None if not base else float(min(1.0 / q, 1.0 / base))


def discriminate(pairs: pl.DataFrame) -> pl.DataFrame:
    """One row per need: within-day AUC, pooled AUC, PR-AUC, lift, and what was scored."""
    rows = []
    for need in NEEDS:
        part = pairs.filter(pl.col("need") == need)
        if not part.height:
            continue
        n_events = int(part["y"].sum())
        rows.append({
            "need": need,
            "within_day_auc": within_day_auc(part),
            "pooled_auc": auc(part["p_mean"], part["y"]),
            "pr_auc": average_precision(part["p_mean"], part["y"]),
            "scaled_brier": scaled_brier(part["p_mean"], part["y"]),
            "brier": brier(part["p_mean"], part["y"]),
            "lift_at_1pct": lift(part, LIFT_QUANTILES[0]),
            "lift_at_10pct": lift(part, LIFT_QUANTILES[1]),
            "lift_ceiling_1pct": lift_ceiling(part, LIFT_QUANTILES[0]),
            "base_rate": n_events / part.height,
            "n": part.height,
            "n_events": n_events,
            # Days that could be scored, and days there were. Dropping eventless days is
            # right but it conditions on the outcome, so the denominator ships with it.
            "n_days": len(daily_auc(part)),
            "n_days_total": part["date"].n_unique(),
        })
    return pl.DataFrame(rows, schema=DISCRIMINATION_SCHEMA)


def run(scores: pl.DataFrame, outcomes: pl.DataFrame, *,
        dates: Sequence[date] | None = None) -> pl.DataFrame:
    """Scores + outcomes -> the discrimination table. The one call the report assembler makes."""
    return discriminate(paired(scores, outcomes, dates=None if dates is None else list(dates)))


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

NEED_LABELS = {"breathing": "Breathing", "heat": "Heat", "mental": "Mental health",
               "treatment_gap": "Treatment gap", "access_loss": "Access loss"}
#: Categorical slots 1-2; validated for adjacent-pair CVD separation on this surface.
WITHIN, POOLED = "#2a78d6", "#c3c2b7"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"


def auc_chart(got: pl.DataFrame) -> go.Figure:
    """A dumbbell per need: pooled AUC, within-day AUC, and the day effect between them.

    Both ends always drawn. Pooled alone is the number that flatters, and a chart that showed
    only the headline would be the same omission this module was written to fix.
    """
    rows = [r for r in got.to_dicts() if r["within_day_auc"] is not None]
    rows.sort(key=lambda r: r["within_day_auc"])
    labels = [NEED_LABELS.get(r["need"], r["need"]) for r in rows]

    fig = go.Figure()
    fig.add_scatter(  # the connector, drawn first so the markers sit on top of it
        x=[v for r in rows for v in (r["within_day_auc"], r["pooled_auc"], None)],
        y=[v for label in labels for v in (label, label, None)],
        mode="lines", line={"color": AXIS, "width": 3}, hoverinfo="skip", showlegend=False)
    fig.add_vline(x=0.5, line={"color": MUTED, "width": 1, "dash": "dot"},
                  annotation={"text": "coin flip", "font": {"color": MUTED, "size": 11}},
                  annotation_position="top")
    for key, name, colour in (("pooled_auc", "Pooled AUC (includes the day effect)", POOLED),
                              ("within_day_auc", "Within-day AUC (the call-list number)",
                               WITHIN)):
        fig.add_scatter(
            x=[r[key] for r in rows], y=labels, mode="markers+text", name=name,
            marker={"size": 15, "color": colour, "line": {"color": SURFACE, "width": 2}},
            text=[f"{r[key]:.3f}" for r in rows],
            textposition="top center" if key == "within_day_auc" else "bottom center",
            textfont={"color": INK if key == "within_day_auc" else MUTED, "size": 11},
            customdata=[(r["pr_auc"] or 0.0, r["base_rate"], r["lift_at_1pct"] or 0.0,
                         r["n_days"]) for r in rows],
            hovertemplate=(f"<b>%{{y}}</b> · {name}<br>%{{x:.3f}}<br>"
                           "PR-AUC %{customdata[0]:.3f} at a %{customdata[1]:.2%} base rate"
                           "<br>top 1% of the day is %{customdata[2]:.1f}x the base rate"
                           "<br>%{customdata[3]} scoreable days<extra></extra>"),
        )
    fig.update_layout(
        title={"text": "Discrimination: who does the model put at the top of today's list?",
               "subtitle": {"text": ("Held-out window · within-day AUC is the mean over days "
                                     "of that day's own AUC · simulated outcomes, synthetic "
                                     "cohort"),
                            "font": {"color": INK_2, "size": 13}},
               "font": {"color": INK, "size": 18}, "x": 0.02, "xanchor": "left"},
        width=760, height=460, margin={"l": 120, "r": 40, "t": 96, "b": 64},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": INK_2},
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": -0.16,
                "font": {"color": INK_2}},
        xaxis={"title": {"text": "AUC", "font": {"color": INK_2, "size": 12}},
               "range": [0.45, 1.0], "gridcolor": GRID, "gridwidth": 1,
               "zerolinecolor": AXIS, "linecolor": AXIS, "tickfont": {"color": MUTED}},
        yaxis={"showgrid": False, "linecolor": SURFACE, "tickfont": {"color": INK_2}},
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": INK}},
    )
    return fig


def write_outputs(got: pl.DataFrame, out_dir: Path = REPORT) -> tuple[Path, Path]:
    """report/discrimination.csv (one row per need) and .html (Plotly, works offline)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, html = out_dir / "discrimination.csv", out_dir / "discrimination.html"
    got.select(list(DISCRIMINATION_SCHEMA)).write_csv(csv)
    # include_plotlyjs=True inlines plotly.js: bigger file, but it opens with the wifi off.
    auc_chart(got).write_html(html, include_plotlyjs=True, full_html=True)
    return csv, html


def _cell(value: float | None, width: int, spec: str) -> str:
    """A number, or a dash wide enough to keep the column straight. `None` means undefined
    rather than zero, and the console has to show the difference."""
    return f"{'--':>{width}}" if value is None else f"{value:{width}{spec}}"


def main() -> int:
    got = run(schema.read("scores"), schema.read("outcomes"))
    csv, html = write_outputs(got)
    print(f"  {'need':14s} {'within-day':>10s} {'pooled':>8s} {'skill':>7s} {'PR-AUC':>8s} "
          f"{'lift@1%':>16s} {'base':>7s} {'days':>7s}")
    for r in got.sort("within_day_auc", descending=True, nulls_last=True).to_dicts():
        ceiling = _cell(r["lift_ceiling_1pct"], 5, ".1f")
        print(f"  {r['need']:14s} {_cell(r['within_day_auc'], 10, '.3f')} "
              f"{_cell(r['pooled_auc'], 8, '.3f')} {_cell(r['scaled_brier'], 7, '.3f')} "
              f"{_cell(r['pr_auc'], 8, '.3f')} "
              f"{_cell(r['lift_at_1pct'], 7, '.1f')}x of {ceiling}x "
              f"{r['base_rate']:7.2%} {r['n_days']:3d}/{r['n_days_total']:<3d}")
    print("  skill is the Brier skill score: a constant at the base rate scores 0.000 here, "
          "where it scores a perfect ECE.")
    print("  within-day is the unweighted mean of per-day c-statistics -- the number that "
          "justifies a call list.")
    print("  pooled is partly 'was today a heat wave'. PR-AUC is secondary and "
          "prevalence-dependent; AUROC is the headline.")
    print("  days is scoreable/total: a day on which the need never happened has no ranking "
          "to score.")
    print(f"wrote {csv.relative_to(schema.ROOT)} and {html.relative_to(schema.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

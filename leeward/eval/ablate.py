"""Ablations -- what each block of the model is actually worth. SPEC §11.

"Does the SiteDown term matter?" currently has a story for an answer: Sandy, and a join
against the evacuation-zone table. This makes it a measurement. For each named block of
coefficients: zero that block, re-score the held-out window, re-run the same calibration and
the same allocator, and report what the care team lost.

At rung 0 that is cheap. The linear predictor is `X @ B`, so dropping a term is zeroing its
rows of `B` -- no refit, no MCMC, no second design matrix. Five ablations plus the full
model over thirty days of the real cohort is a couple of dozen seconds.

    python -m leeward.eval.ablate            # -> report/ablations.{csv,json,html}
    python -m leeward.eval.ablate --k 20     # a tighter call budget

**A block is dropped at the coefficient, not at the input.** Setting `site_down` to False in
the hazard tables would drop four terms at once and change the design every other term is
measured against; zeroing rows of `B` leaves `X` exactly as it was. That is also what makes
the honest half of the test possible: on a day the block was not active its columns of `X`
are zero, so the ablated model *is* the full model, bit for bit, and any difference there is
a bug rather than a finding.

Two numbers per row, both already defined elsewhere in this package rather than re-invented
here: the pooled ECE from `calibration.py` over the same held-out window, and mean harm
averted per day from `decision_quality.py` under a K-call budget, allocated by the same
`leeward.decision.allocate` the Impact chart uses. Each model is binned on **its own**
equal-mass bins, because a weaker predictor has a different risk distribution and forcing it
into the full model's bins would report the shift rather than the miscalibration.

The scoring loop below is `model/score_prior.py`'s, with a mask on `B`. It is a deliberate
copy rather than a call, because `score()` draws its own coefficients and there is nowhere to
hand it different ones; `tests/test_ablate.py` pins the copy by scoring an empty ablation and
asserting the frame is identical to `score_prior.score()`'s. If the model lane changes the
scorer, that test goes red rather than this table quietly comparing two different models.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from leeward import schema
from leeward.eval import calibration as cal
from leeward.eval import decision_quality as dq
from leeward.model import design, priors, score_prior
from leeward.schema import NEEDS

REPORT = schema.ROOT / "report"

HOLDOUT_DAYS = cal.HOLDOUT_DAYS
K_CALLS = 40                      # `AblationRow.harm_averted_at_40`; SPEC §11's middle budget
N_DRAWS = score_prior.N_DRAWS
SEED = score_prior.SEED

#: The row every other row is measured against.
FULL = "none (full model)"

TABLE_SCHEMA = {
    "dropped": pl.Utf8, "n_features": pl.Int32, "ece": pl.Float64,
    "harm_averted_at_k": pl.Float64, "d_ece": pl.Float64, "d_harm_averted": pl.Float64,
    "claim": pl.Utf8,
}
DAILY_SCHEMA = {"dropped": pl.Utf8, "date": pl.Date, "harm_averted": pl.Float64}

_K = design.K
_THREADS = min(8, os.cpu_count() or 1)
_TERM_NAMES = [t.name for t in design.TERMS]
_DRIVER_TERMS = [j for j, name in enumerate(_TERM_NAMES) if name != "alpha"]
_PHRASES = [design.TERMS[j].phrase for j in _DRIVER_TERMS]
_FEATURE_INDEX = {f: i for i, f in enumerate(design.FEATURES)}


# --------------------------------------------------------------------------- #
# The named set of blocks
# --------------------------------------------------------------------------- #

def term_features(*names: str) -> tuple[str, ...]:
    """Every feature column belonging to these terms -- a lag curve included."""
    out: list[str] = []
    for name in names:
        term = design.TERM_BY_NAME.get(name)
        if term is None:
            raise ValueError(f"{name!r} is not a design term; "
                             f"known terms: {sorted(design.TERM_BY_NAME)}")
        out.extend(term.feature_names)
    return tuple(out)


def lag_features(name: str, first: int) -> tuple[str, ...]:
    """A lag curve from `first` on, leaving the same-day column in place."""
    term = design.TERM_BY_NAME[name]
    if not term.lagged:
        raise ValueError(f"{name!r} is not a lagged term")
    return term.feature_names[first:]


#: Medication terms, derived from the design rather than listed, so a term that lands in a
#: later round joins the block without anyone remembering to add it here. `mail_order` and
#: `days_supply` are in because what they carry is a fill running out, which is medication.
_MED_PREFIX = "med_"
_MED_EXACT = frozenset({"acb_score", "mail_order_pharmacy", "days_supply_remaining"})


def _reads_medication(term: design.Term) -> bool:
    roots = {c for e in term.exprs for c in e.meta.root_names()}
    return any(r.startswith(_MED_PREFIX) or r in _MED_EXACT for r in roots)


#: term name -> the plain-language phrase, for the terms the medication block covers.
MEDICATION_TERMS: tuple[tuple[str, str], ...] = tuple(
    (t.name, t.phrase) for t in design.TERMS if _reads_medication(t))


@dataclass(frozen=True)
class Ablation:
    """One block of coefficients to zero, and the claim it is there to support."""
    key: str
    #: What lands in `AblationRow.dropped`. Unique: it is the table's key.
    label: str
    features: tuple[str, ...]
    claim: str


#: Every term that is zero on a calm day: the hazards, their lag curves and the interactions
#: built on them. Dropping the lot leaves a static risk score that has never heard of weather.
CLIMATE_TERMS: tuple[str, ...] = tuple(t.name for t in design.TERMS if t.kind == "hazard")

ABLATIONS: tuple[Ablation, ...] = (
    Ablation("climate", "every climate term (hazards, lags, interactions)",
             term_features(*CLIMATE_TERMS),
             "the whole premise -- does the weather change who the team calls?"),
    Ablation("psi_sitedown", "psi_sitedown",
             term_features("psi_sitedown"),
             "every patient of a closed VA station is harder to treat"),
    Ablation("theta_sitedown_x_sitedependent", "theta_sitedown_x_sitedependent",
             term_features("theta_sitedown_x_sitedependent"),
             "dialysis, infusion and methadone patients of a closed station, worst of all"),
    Ablation("sitedown", "the whole SiteDown block",
             term_features("psi_sitedown", "theta_sitedown_x_sitedependent",
                           "theta_sitedown_x_controlled"),
             "everything Leeward claims about a station closing -- the never-cut term"),
    Ablation("medications", "the medication block",
             term_features(*(name for name, _ in MEDICATION_TERMS)),
             "CDC's heat-risk drug classes, cold chain, controlled substances and supply"),
    Ablation("heat_lags", "heat lags 1-3 (same-day heat kept)",
             lag_features("delta_heat", 1),
             "heat harm that lands in the days after the peak, not on it"),
)


# --------------------------------------------------------------------------- #
# Scoring with a block zeroed. `model/score_prior.py`'s loop, plus a mask on B.
# --------------------------------------------------------------------------- #

def drop_rows(drop: Sequence[str]) -> np.ndarray:
    """Feature names -> their rows of B. Refuses a name the design does not have."""
    unknown = [f for f in drop if f not in _FEATURE_INDEX]
    if unknown:
        raise ValueError(
            f"{unknown} is not a design feature, so zeroing it would ablate nothing and the "
            f"row would claim the block is worthless. Features are {design.FEATURES}.")
    return np.array([_FEATURE_INDEX[f] for f in drop], dtype=int)


def _summarise(d: design.Design, B_flat: np.ndarray, B_mean: np.ndarray,
               n_draws: int) -> pl.DataFrame:
    """score_prior._summarise. See the module docstring for why this is a copy."""
    n = d.X.shape[0]
    mean, lo, hi, share = (np.empty((n, _K)) for _ in range(4))

    def block(s: int) -> None:
        e = min(s + score_prior.BLOCK_ROWS, n)
        eta = (d.X[s:e] @ B_flat).reshape(e - s, _K, n_draws)
        np.clip(eta, -score_prior.ETA_CLIP, score_prior.ETA_CLIP, out=eta)
        p = 1.0 / (1.0 + np.exp(-eta))
        m = p.mean(axis=-1)
        q = np.quantile(p, [score_prior.LO_Q, score_prior.HI_Q], axis=-1)
        mean[s:e] = m
        lo[s:e] = np.minimum(q[0], m)
        hi[s:e] = np.maximum(q[1], m)
        share[s:e] = np.clip(p.var(axis=-1) / (m * (1.0 - m)), 0.0, 1.0)

    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        list(pool.map(block, range(0, n, score_prior.BLOCK_ROWS)))

    contrib = design.term_contributions(d.X, B_mean)[:, _DRIVER_TERMS, :]
    contrib = contrib.transpose(0, 2, 1).reshape(n * _K, -1)
    top = np.argsort(-np.abs(contrib), axis=1, kind="stable")[:, :3]
    val = np.take_along_axis(contrib, top, axis=1)
    keep = np.abs(val) >= score_prior.MIN_DRIVER_LOGODDS
    code = np.where(keep, top, -1)

    out = {
        "_i": pl.Series(np.repeat(d.vet_index, _K), dtype=pl.UInt32),
        "date": pl.Series(np.repeat(d.date, _K), dtype=pl.Date),
        "_k": pl.Series(np.tile(np.arange(_K), n), dtype=pl.Int8),
        "p_mean": mean.ravel(),
        "p_lo80": lo.ravel(),
        "p_hi80": hi.ravel(),
        "p_epistemic_share": share.ravel(),
    }
    for i in range(3):
        out[f"_d{i + 1}"] = pl.Series(code[:, i], dtype=pl.Int32)
        out[f"driver_{i + 1}_contrib"] = pl.Series(np.where(keep[:, i], val[:, i], np.nan),
                                                   nan_to_null=True)
    return pl.DataFrame(out)


def _decode(col: str, labels: list[str]) -> pl.Expr:
    return pl.col(col).replace_strict(list(range(len(labels))), labels, default=None,
                                      return_dtype=pl.Utf8)


def score_ablated(cohort: pl.DataFrame, hazards: pl.DataFrame, site_status: pl.DataFrame, *,
                  drop: Sequence[str] = (), dates: Sequence[date] | None = None,
                  n_draws: int = N_DRAWS, seed: int = SEED) -> pl.DataFrame:
    """The `scores` table with `drop`'s rows of B held at zero. `drop=()` is the full model.

    The coefficients are drawn once from `seed`, so two ablations differ only where their
    blocks differ -- re-drawing per ablation would move every row of every day.
    """
    dates = sorted(hazards["date"].unique().to_list()) if dates is None else sorted(dates)
    ref = hazards["date"].min()

    rows = drop_rows(drop)
    B = priors.draw(n_draws, seed=seed)
    if rows.size:
        B = B.copy()
        B[:, rows, :] = 0.0
    B_flat = np.ascontiguousarray(B.transpose(1, 2, 0).reshape(design.P, _K * n_draws))
    B_mean = B.mean(axis=0)

    parts = []
    for i in range(0, len(dates), score_prior.DAYS_PER_BUILD):
        d = design.build(cohort, hazards, site_status,
                         dates=dates[i:i + score_prior.DAYS_PER_BUILD], ref_date=ref)
        parts.append(_summarise(d, B_flat, B_mean, n_draws))

    if not parts:
        return schema.empty("scores")
    df = pl.concat(parts)
    return df.select(
        cohort["veteran_id"].gather(df["_i"]),
        "date",
        _decode("_k", NEEDS).alias("need"),
        "p_mean", "p_lo80", "p_hi80", "p_epistemic_share",
        *[_decode(f"_d{i}", _PHRASES).alias(f"driver_{i}") for i in (1, 2, 3)],
        *[f"driver_{i}_contrib" for i in (1, 2, 3)],
        model_rung=pl.lit(score_prior.RUNG, dtype=pl.Int32),
    )


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #

def holdout_dates(hazards: pl.DataFrame, outcomes: pl.DataFrame,
                  n_days: int = HOLDOUT_DAYS) -> list[date]:
    """The last `n_days` days with both a hazard row and an outcome: the same window
    `calibration.py` and `decision_quality.py` hold out, chosen before anything is scored."""
    both = set(hazards["date"].unique().to_list()) & set(outcomes["date"].unique().to_list())
    if not both:
        raise ValueError("hazards and outcomes share no dates; nothing to ablate over")
    return sorted(both)[-n_days:]


def _leeward_daily(scores: pl.DataFrame, cohort: pl.DataFrame, outcomes: pl.DataFrame, *,
                   w, tau, k: int, dates: Sequence[date]) -> pl.DataFrame:
    """Harm averted per day by `allocate` under a K-call budget: one row per day.

    This is `decision_quality.evaluate`'s Leeward column, through its own two functions, with
    the three baselines left out -- they do not read the model, so re-running them once per
    ablation would be three quarters of the wall clock spent re-deriving the same numbers.
    `tests/test_ablate.py` checks the answer against `evaluate` itself.
    """
    from leeward.decision.allocate import allocate

    rows = []
    for day in dates:
        scores_day = scores.filter(pl.col("date") == day)
        acts = dq.leeward_actions(allocate, scores_day, cohort, k)
        rows.append({"date": day,
                     "harm_averted": dq.harm_averted(
                         acts, outcomes.filter(pl.col("date") == day), w, tau)})
    return pl.DataFrame(rows, schema={"date": pl.Date, "harm_averted": pl.Float64})


@dataclass(frozen=True)
class Result:
    """The table the report serves, and the per-day frame it was summarised from."""
    table: pl.DataFrame     # one row per dropped block, the full model first
    daily: pl.DataFrame     # dropped x date, harm averted by the allocator that day
    k: int
    dates: tuple[date, ...]


def run(cohort: pl.DataFrame, hazards: pl.DataFrame, site_status: pl.DataFrame,
        outcomes: pl.DataFrame, *, dates: Sequence[date] | None = None,
        ablations: Sequence[Ablation] = ABLATIONS, k: int = K_CALLS,
        n_draws: int = N_DRAWS, seed: int = SEED) -> Result:
    """Score the full model and every ablation over one window, and measure both."""
    dates = holdout_dates(hazards, outcomes) if dates is None else sorted(dates)
    w, tau = dq.decision_weights()

    models: list[tuple[str, tuple[str, ...], str]] = [
        (FULL, (), "every term the model has"),
        *((a.label, a.features, a.claim) for a in ablations)]
    seen = [label for label, _, _ in models]
    if len(set(seen)) != len(seen):
        raise ValueError(f"two ablations share a `dropped` label, which is the table's key: "
                         f"{sorted(seen)}")

    daily, rows = [], []
    for label, drop, claim in models:
        # One scores frame alive at a time: thirty days of the real cohort is 1.5M rows, and
        # six of them at once is most of a laptop.
        scores = score_ablated(cohort, hazards, site_status, drop=drop, dates=dates,
                               n_draws=n_draws, seed=seed)
        reliability = cal.run(scores, outcomes, dates=dates)
        lee = _leeward_daily(scores, cohort, outcomes, w=w, tau=tau, k=k, dates=dates) \
            .select(pl.lit(label, dtype=pl.Utf8).alias("dropped"), "date", "harm_averted")
        daily.append(lee)
        rows.append({"dropped": label, "n_features": len(drop),
                     "ece": cal.ece_overall(reliability),
                     "harm_averted_at_k": float(lee["harm_averted"].mean()),
                     "claim": claim})

    table = pl.DataFrame(rows, schema={c: t for c, t in TABLE_SCHEMA.items()
                                       if c not in ("d_ece", "d_harm_averted")})
    base = table.row(0, named=True)
    table = table.with_columns(
        (pl.col("ece") - base["ece"]).alias("d_ece"),
        (pl.col("harm_averted_at_k") - base["harm_averted_at_k"]).alias("d_harm_averted"),
    ).select(list(TABLE_SCHEMA))
    return Result(table=table, daily=pl.concat(daily).select(list(DAILY_SCHEMA)),
                  k=k, dates=tuple(dates))


def rows(table: pl.DataFrame) -> list[dict]:
    """`AblationRow`'s shape, for `report.json` and the Model report screen."""
    return table.select("dropped", "ece",
                        pl.col("harm_averted_at_k").alias("harm_averted_at_40")).to_dicts()


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

COST, GAIN = "#eb6834", "#1baf7a"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"


def bar_chart(result: Result) -> go.Figure:
    """What each block is worth: the harm averted that goes away when it is dropped."""
    dropped = result.table.filter(pl.col("dropped") != FULL).sort("d_harm_averted")
    loss = [-v for v in dropped["d_harm_averted"]]
    first, last = result.dates[0], result.dates[-1]
    # The bar labels sit outside the bar end, so both ends of the axis need room for one --
    # otherwise a negative bar's label lands on top of the block names in the left margin.
    span = max((abs(v) for v in loss), default=1.0) or 1.0
    fig = go.Figure()
    fig.add_bar(
        x=loss, y=dropped["dropped"].to_list(), orientation="h",
        marker={"color": [COST if v > 0 else GAIN for v in loss], "cornerradius": 4},
        text=[f"{v:+.2f}" for v in loss], textposition="outside", cliponaxis=False,
        textfont={"color": INK},
        customdata=list(zip(dropped["ece"], dropped["d_ece"], strict=True)),
        hovertemplate=("<b>drop %{y}</b><br>%{x:.2f} less harm averted per day<br>"
                       "ECE %{customdata[0]:.4f} (%{customdata[1]:+.4f})<extra></extra>"),
    )
    fig.update_layout(
        title={"text": "What each block of the model is worth",
               "subtitle": {"text": (f"Harm averted per day lost when the block is dropped · "
                                     f"{result.k} calls a day · rung 0, prior-only<br>"
                                     f"Held-out days {first} to {last} · a negative bar is a "
                                     f"block the care team is better off without"),
                            "font": {"color": INK_2, "size": 13}},
               "font": {"color": INK, "size": 18}, "x": 0.02, "xanchor": "left"},
        width=940, height=140 + 54 * max(dropped.height, 1),
        margin={"l": 300, "r": 40, "t": 118, "b": 56},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": INK_2},
        xaxis={"title": {"text": "Severity-weighted events averted per day, forgone",
                         "font": {"color": INK_2, "size": 12}},
               "range": [-span * 1.35, span * 1.35],
               "gridcolor": GRID, "zerolinecolor": AXIS, "tickfont": {"color": MUTED}},
        yaxis={"autorange": "reversed", "showgrid": False, "linecolor": AXIS,
               "tickfont": {"color": MUTED}},
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": INK}},
    )
    return fig


def write_outputs(result: Result, out_dir: Path = REPORT) -> tuple[Path, Path, Path]:
    """ablations.csv (the whole table), ablations.json (the report payload) and the chart."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, js, html = (out_dir / f"ablations.{ext}" for ext in ("csv", "json", "html"))
    result.table.select(list(TABLE_SCHEMA)).write_csv(csv)
    js.write_text(json.dumps({
        "k": result.k,
        "dates": [str(d) for d in (result.dates[0], result.dates[-1])],
        "ablations": rows(result.table),
    }, indent=2) + "\n", encoding="utf-8")
    # include_plotlyjs=True inlines plotly.js: bigger file, but it opens with the wifi off.
    bar_chart(result).write_html(html, include_plotlyjs=True, full_html=True)
    return csv, js, html


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=HOLDOUT_DAYS,
                    help="length of the held-out window (default: the last 30 scored days)")
    ap.add_argument("--k", type=int, default=K_CALLS, help="calls a day (default 40)")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    cohort, hazards = schema.read("cohort"), schema.read("hazards")
    sites, outcomes = schema.read("site_status"), schema.read("outcomes")
    dates = holdout_dates(hazards, outcomes, n_days=args.days)

    t0 = time.perf_counter()
    result = run(cohort, hazards, sites, outcomes, dates=dates, k=args.k,
                 n_draws=args.draws, seed=args.seed)
    written = write_outputs(result)
    secs = time.perf_counter() - t0

    print(f"rung {score_prior.RUNG} · {len(ABLATIONS) + 1} models × {cohort.height:,} "
          f"veterans × {len(dates)} days ({dates[0]} .. {dates[-1]}) · {args.k} calls a day · "
          f"{args.draws} draws, seed {args.seed} · {secs:.1f}s")
    with pl.Config(tbl_rows=len(ABLATIONS) + 2, tbl_width_chars=170, fmt_str_lengths=60):
        print(result.table.select("dropped", "n_features", "ece", "d_ece",
                                  "harm_averted_at_k", "d_harm_averted"))
    if args.k != K_CALLS:
        print(f"  NOTE: the report contract's field is `harm_averted_at_40`, so the Model "
              f"report screen will label these {args.k}-call numbers as 40. report/"
              f"ablations.json records the real budget under `k`.")
    for row in result.table.filter(pl.col("dropped") != FULL).to_dicts():
        # Exactly zero on both numbers means the block's columns of X were zero all window:
        # it never fired, which is a different statement from "it fired and was worthless".
        if row["d_harm_averted"] == 0.0 and row["d_ece"] == 0.0:
            print(f"  NOTE: {row['dropped']} was never active in this window -- every scored "
                  f"row is identical to the full model's, so this row measures nothing. "
                  f"Choose a window the block actually fires in.")
        elif row["d_harm_averted"] >= 0:
            print(f"  NOTE: dropping {row['dropped']} cost the care team nothing "
                  f"({row['d_harm_averted']:+.2f} harm averted a day). At this rung that "
                  f"block is not earning its place, and the deck should not claim it does.")
    print("wrote " + ", ".join(str(p.relative_to(schema.ROOT)) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

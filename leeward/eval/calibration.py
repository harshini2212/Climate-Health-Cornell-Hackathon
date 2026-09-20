"""Calibration -- when Leeward says 8 percent, does it happen 8 percent of the time? SPEC §11.

On the held-out window, every scored veteran-day-need is paired with what actually happened
and the pairs are grouped into bins of similar predicted risk. A bin's `predicted` is the
mean risk in it, its `observed` is the fraction that happened, and

    ECE = Σ_bins (n_bin / N) · |observed - predicted|

per need. SPEC §11's bar is ECE < 0.03.

**The bins hold equal numbers of rows, not equal widths.** This is the one decision in the
module worth arguing about, so: a daily hazard is a small number. The real rung-0 run
predicts well under a percent on most veteran-days, so ten equal-width bins across [0, 1]
would put 99 percent of the rows in the first bin and report an ECE near zero however wrong
the model was -- the curve would be a dot in the corner and the number would be a lie of
resolution. Equal-mass bins put the resolution where the predictions are. Ties are held
together in one bin, so a constant predictor gives one bin rather than ten copies of itself.

The fairness audit reuses `bin_edges` so that every group is scored on the whole window's
bins; a per-group re-bin would make two groups' ECEs incomparable.

    python -m leeward.eval.calibration     # -> report/calibration.{csv,html}
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from leeward import schema
from leeward.schema import NEEDS

REPORT = schema.ROOT / "report"

HOLDOUT_DAYS = 30                 # days 91-120 of the 120-day simulation
BINS = 10
ECE_BAR = 0.03                    # SPEC §11

RELIABILITY_SCHEMA = {"need": pl.Utf8, "bin": pl.Int32, "predicted": pl.Float64,
                      "observed": pl.Float64, "n": pl.Int32}

PAIRED_COLUMNS = ["veteran_id", "date", "need", "p_mean", "y"]

Edges = Mapping[str, np.ndarray]


# --------------------------------------------------------------------------- #
# The held-out window
# --------------------------------------------------------------------------- #

def holdout_dates(scores: pl.DataFrame, outcomes: pl.DataFrame,
                  n_days: int = HOLDOUT_DAYS) -> list[date]:
    """The last `n_days` dates that have both scores and outcomes: days 91-120 of a
    120-day run; every day of a shorter fixture."""
    both = set(scores["date"].unique().to_list()) & set(outcomes["date"].unique().to_list())
    if not both:
        raise ValueError("scores and outcomes share no dates; nothing to evaluate")
    return sorted(both)[-n_days:]


def paired(scores: pl.DataFrame, outcomes: pl.DataFrame, *,
           dates: Sequence[date] | None = None) -> pl.DataFrame:
    """Every scored veteran-day-need in the window, next to what happened.

    A scored row with no outcome is an error. Filling it with zero would say "nothing
    happened" exactly where the simulation has nothing to say, and every such row would
    flatter the model.
    """
    dates = holdout_dates(scores, outcomes) if dates is None else list(dates)
    left = (scores.filter(pl.col("date").is_in(dates))
                  .select("veteran_id", "date", "need", "p_mean"))
    joined = left.join(outcomes.select("veteran_id", "date", "need", "y"),
                       on=["veteran_id", "date", "need"], how="left")
    ghosts = joined.filter(pl.col("y").is_null())
    if ghosts.height:
        example = ghosts.select("veteran_id", "date", "need").row(0)
        raise ValueError(
            f"{ghosts.height} scored veteran-day-needs have no outcome, e.g. {example}. "
            "Scoring them as 0 would quietly flatter the model where it is blind.")
    return joined.select(PAIRED_COLUMNS)


# --------------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------------- #

def bin_edges(pairs: pl.DataFrame, *, n_bins: int = BINS) -> dict[str, np.ndarray]:
    """Interior bin boundaries per need: the `n_bins - 1` quantiles of that need's risks.

    `np.unique` collapses repeated quantiles, so a predictor with fewer distinct values
    than bins yields fewer bins rather than a row of empty ones.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2; got {n_bins}")
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    out: dict[str, np.ndarray] = {}
    for (need,), part in pairs.group_by("need"):
        p = part["p_mean"].to_numpy()
        out[str(need)] = np.unique(np.quantile(p, qs)) if p.size else np.empty(0)
    return out


def assign_bins(pairs: pl.DataFrame, edges: Edges) -> pl.DataFrame:
    """`pairs` plus a `bin` column: which of that need's bins each row falls in.

    Exposed because the fairness audit bins once on the whole window and then aggregates
    by group -- binning inside each group would mean each group had different bins.
    """
    parts = []
    for need in NEEDS:
        part = pairs.filter(pl.col("need") == need)
        if not part.height:
            continue
        if need not in edges:
            raise ValueError(f"no bin edges for need {need!r}; have {sorted(edges)}")
        idx = np.searchsorted(np.asarray(edges[need]), part["p_mean"].to_numpy(), side="left")
        parts.append(part.with_columns(pl.Series("bin", idx, dtype=pl.Int32)))
    return pl.concat(parts) if parts else pairs.with_columns(
        pl.lit(None, dtype=pl.Int32).alias("bin"))


def reliability(pairs: pl.DataFrame, *, edges: Edges | None = None,
                n_bins: int = BINS) -> pl.DataFrame:
    """One row per (need, bin): predicted, observed and how many rows fell in it.

    Pass `edges` to bin against someone else's boundaries -- what the fairness audit does
    so that a group's ECE is comparable with the cohort's.
    """
    if pairs.is_empty():
        return pl.DataFrame(schema=RELIABILITY_SCHEMA)
    edges = bin_edges(pairs, n_bins=n_bins) if edges is None else edges
    return (assign_bins(pairs, edges)
              .group_by("need", "bin")
              .agg(pl.col("p_mean").mean().alias("predicted"),
                   pl.col("y").mean().cast(pl.Float64).alias("observed"),
                   pl.len().cast(pl.Int32).alias("n"))
              .sort(pl.col("need").cast(pl.Enum(NEEDS)), "bin")
              .select(list(RELIABILITY_SCHEMA)))


# --------------------------------------------------------------------------- #
# ECE
# --------------------------------------------------------------------------- #

_GAP = pl.col("n") * (pl.col("observed") - pl.col("predicted")).abs()


def ece(rel: pl.DataFrame) -> dict[str, float]:
    """Expected calibration error per need, weighted by how many rows each bin holds."""
    per = (rel.with_columns(_GAP.alias("_gap"))
              .group_by("need")
              .agg((pl.col("_gap").sum() / pl.col("n").sum()).alias("ece")))
    got = {r["need"]: float(r["ece"]) for r in per.to_dicts()}
    return {need: got[need] for need in NEEDS if need in got}


def ece_overall(rel: pl.DataFrame) -> float:
    """One ECE across every need, pooled by row count. The fairness audit's per-group number."""
    if rel.is_empty() or rel["n"].sum() == 0:
        return 0.0
    return float(rel.select(_GAP.sum() / pl.col("n").sum()).item())


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

#: Categorical slots 1-5, validated for adjacent-pair CVD separation on this surface.
NEED_COLORS = {"breathing": "#2a78d6", "heat": "#eb6834", "mental": "#1baf7a",
               "treatment_gap": "#eda100", "access_loss": "#e87ba4"}
NEED_LABELS = {"breathing": "Breathing", "heat": "Heat", "mental": "Mental health",
               "treatment_gap": "Treatment gap", "access_loss": "Access loss"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"


def reliability_curve(rel: pl.DataFrame) -> go.Figure:
    """Predicted against observed, one line per need, with the perfectly-calibrated diagonal.

    The axes are log-scaled: rung-0 risks span two orders of magnitude, and on a linear axis
    every bin but the last would sit on top of the origin.
    """
    by_need = ece(rel)
    # A log axis has no room for zero, and an all-zero bin is ordinary at the bottom of a
    # rare-event curve. Frame the window on the smallest positive value on either axis.
    positive = [v for v in rel["predicted"].to_list() + rel["observed"].to_list() if v > 0]
    lo = max(1e-5, (min(positive) if positive else 1e-3) * 0.7)
    hi = min(1.0, (max(positive) if positive else 1.0) * 1.4)

    fig = go.Figure()
    fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect",
                    line={"color": AXIS, "width": 2, "dash": "dot"},
                    hoverinfo="skip", showlegend=False)
    for need in NEEDS:
        s = rel.filter(pl.col("need") == need)
        if not s.height:
            continue
        fig.add_scatter(
            x=s["predicted"].to_list(), y=s["observed"].to_list(),
            mode="lines+markers", name=f"{NEED_LABELS[need]} · ECE {by_need.get(need, 0):.3f}",
            line={"color": NEED_COLORS[need], "width": 2},
            marker={"size": 9, "line": {"color": SURFACE, "width": 2}},
            customdata=s["n"].to_list(),
            hovertemplate=(f"<b>{NEED_LABELS[need]}</b><br>predicted %{{x:.3f}}<br>"
                           "observed %{y:.3f}<br>%{customdata:,} veteran-days<extra></extra>"),
        )
    axis = {"type": "log", "range": [np.log10(lo), np.log10(hi)], "gridcolor": GRID,
            "gridwidth": 1, "zerolinecolor": AXIS, "linecolor": AXIS,
            "tickfont": {"color": MUTED}, "tickformat": ".1%"}
    fig.update_layout(
        title={"text": "Reliability: predicted risk against what happened",
               "subtitle": {"text": (f"Held-out window · equal-mass bins · pooled ECE "
                                     f"{ece_overall(rel):.3f} · simulated outcomes, "
                                     "synthetic cohort"),
                            "font": {"color": INK_2, "size": 13}},
               "font": {"color": INK, "size": 18}, "x": 0.02, "xanchor": "left"},
        width=760, height=520, margin={"l": 70, "r": 24, "t": 96, "b": 64},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": INK_2},
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": -0.16,
                "font": {"color": INK_2}},
        xaxis={**axis, "title": {"text": "Predicted daily risk",
                                 "font": {"color": INK_2, "size": 12}}},
        yaxis={**axis, "title": {"text": "Observed rate",
                                 "font": {"color": INK_2, "size": 12}}},
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": INK}},
    )
    return fig


def write_outputs(rel: pl.DataFrame, out_dir: Path = REPORT) -> tuple[Path, Path]:
    """report/calibration.csv (one row per bin) and .html (Plotly, works offline)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, html = out_dir / "calibration.csv", out_dir / "calibration.html"
    rel.select(list(RELIABILITY_SCHEMA)).write_csv(csv)
    # include_plotlyjs=True inlines plotly.js: bigger file, but it opens with the wifi off.
    reliability_curve(rel).write_html(html, include_plotlyjs=True, full_html=True)
    return csv, html


def run(scores: pl.DataFrame, outcomes: pl.DataFrame, *,
        dates: Iterable[date] | None = None, n_bins: int = BINS) -> pl.DataFrame:
    """Scores + outcomes -> the reliability table. The one call the report assembler makes."""
    return reliability(paired(scores, outcomes,
                              dates=None if dates is None else list(dates)), n_bins=n_bins)


def main() -> int:
    rel = run(schema.read("scores"), schema.read("outcomes"))
    csv, html = write_outputs(rel)
    for need, value in ece(rel).items():
        print(f"  {need:14s} ECE {value:.4f}  {'ok' if value < ECE_BAR else 'over the bar'}")
    print(f"  {'pooled':14s} ECE {ece_overall(rel):.4f}  (SPEC §11 bar is {ECE_BAR})")
    print(f"wrote {csv.relative_to(schema.ROOT)} and {html.relative_to(schema.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

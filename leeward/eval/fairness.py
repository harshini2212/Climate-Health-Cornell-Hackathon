"""Fairness audit -- who does this model miss? SPEC §11.

Every group in every stratum gets two numbers:

    ECE   how well calibrated the risks are here, against the whole window's bins
    FNR   of the veteran-days where a need actually occurred, the share where nobody
          was put in front of a human

and the comparison that matters, `fnr_ratio_to_cohort`. A group whose ratio exceeds
1 + 20 percent is **flagged**.

Three choices worth stating out loud, because they are the ones that decide what the audit
can see:

1. **A false negative is defined by the shipped rule, not by an eval-only threshold.** A
   veteran-day counts as reached when `leeward.decision.tiers` puts it in `act_now` or
   `find_out` -- the two tiers that cost a real person's attention. Auditing some other
   cutoff would audit a system nobody is running.
2. **Every group is binned on the whole window's edges** (`calibration.bin_edges`). Re-binning
   inside each group would make two groups' ECEs incomparable, which is the quiet way to
   make a calibration gap disappear.
3. **Only the harmed side is flagged.** A group the model misses *less* often than the
   cohort is not a finding; `fnr_ratio_to_cohort` is in the table either way, so a reader can
   see both directions. `n_events` is there too, so a ratio computed off a handful of events
   can be read as the noise it is rather than trusted as a verdict.

No group is ever dropped -- not for being small, not for having no events, not for an
unknown race. A failing audit is displayed, never suppressed; `test_guardrails.py` asserts
this file contains no swallowed exceptions, and it should stay that way.

    python -m leeward.eval.fairness     # -> report/fairness.{csv,html}
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import plotly.graph_objects as go
import polars as pl

from leeward import schema
from leeward.eval import calibration as cal

REPORT = schema.ROOT / "report"

#: A group whose FNR exceeds the cohort's by more than this share is flagged.
FNR_GAP = 0.20

#: The tiers that put a veteran in front of a human. Everything else is a miss.
REACHED_TIERS = ("act_now", "find_out")

UNKNOWN = "unknown"

AUDIT_SCHEMA = {"stratum": pl.Utf8, "group": pl.Utf8, "n": pl.Int32, "n_events": pl.Int32,
                "n_missed": pl.Int32, "ece": pl.Float64, "fnr": pl.Float64,
                "fnr_ratio_to_cohort": pl.Float64, "flagged": pl.Boolean}

#: The columns of `FairnessRow` -- what `report.json` carries. `n_events` and `n_missed`
#: stay in the CSV and on the console; the API contract has no room for them.
REPORT_COLUMNS = ["stratum", "group", "n", "ece", "fnr", "fnr_ratio_to_cohort", "flagged"]


@dataclass(frozen=True)
class Stratum:
    """One way of cutting the cohort, and the label each veteran gets under it."""
    name: str
    expr: pl.Expr
    doc: str


_MEDS = pl.col("n_active_meds")

STRATA: tuple[Stratum, ...] = (
    Stratum("borough", pl.col("borough"), "NYC borough"),
    Stratum("hvi_band", pl.format("HVI {}", pl.col("hvi")),
            "NYC Heat Vulnerability Index, 1 (lowest) to 5 (highest)"),
    Stratum("evac_zone", pl.when(pl.col("evac_zone") == 0).then(pl.lit("no zone"))
                           .otherwise(pl.format("zone {}", pl.col("evac_zone"))),
            "NYC hurricane evacuation zone"),
    Stratum("income_band", pl.col("income_band"), "ACS-derived income band"),
    Stratum("caregiver", pl.col("caregiver"), "Caregiver status, incl. VA PCAFC"),
    # 5 is the usual polypharmacy line and 10 the hyperpolypharmacy one; they are naming
    # conventions rather than measurements, so no number here is a claim about NYC.
    Stratum("medication_burden", pl.when(_MEDS < 5).then(pl.lit("0-4 meds"))
                                   .when(_MEDS < 10).then(pl.lit("5-9 meds"))
                                   .otherwise(pl.lit("10+ meds")),
            "Active medication count"),
    Stratum("race", pl.col("race"), "Fairness audit only; never an input to the model"),
    Stratum("ethnicity", pl.col("ethnicity"), "Fairness audit only"),
)

STRATUM_ORDER = pl.Enum([s.name for s in STRATA])


# --------------------------------------------------------------------------- #
# The two ingredients
# --------------------------------------------------------------------------- #

def strata(cohort: pl.DataFrame) -> pl.DataFrame:
    """veteran_id plus one label column per stratum. A missing value is its own group."""
    return cohort.select(
        "veteran_id",
        *[s.expr.cast(pl.Utf8).fill_null(UNKNOWN).alias(s.name) for s in STRATA])


def reached(scores: pl.DataFrame, weights: dict[str, float] | None = None) -> pl.DataFrame:
    """veteran_id, date, reached -- did the shipped tier rule put this day in front of a human?"""
    from leeward.decision import tiers
    return (tiers.assign(scores, weights)
                 .select("veteran_id", "date",
                         pl.col("tier").is_in(REACHED_TIERS).alias("reached")))


def veteran_days(pairs: pl.DataFrame, scores: pl.DataFrame,
                 weights: dict[str, float] | None = None) -> pl.DataFrame:
    """One row per veteran-day: did a need occur, and was the veteran reached?

    A day with three needs is still one phone call, so it is one row in the FNR denominator.
    """
    needed = (pairs.group_by("veteran_id", "date")
                   .agg((pl.col("y").max() == 1).alias("needed")))
    days = needed.join(reached(scores, weights), on=["veteran_id", "date"], how="left")
    unresolved = days.filter(pl.col("reached").is_null())
    if unresolved.height:
        raise ValueError(
            f"{unresolved.height} veteran-days have an outcome but no tier, e.g. "
            f"{unresolved.select('veteran_id', 'date').row(0)}")
    return days


# --------------------------------------------------------------------------- #
# The audit
# --------------------------------------------------------------------------- #

def _ece_by_group(binned: pl.DataFrame, column: str) -> pl.DataFrame:
    """Pooled ECE per group, on bins that were cut on the whole window."""
    gap = pl.col("n") * (pl.col("observed") - pl.col("predicted")).abs()
    return (binned.group_by(column, "need", "bin")
                  .agg(pl.col("p_mean").mean().alias("predicted"),
                       pl.col("y").mean().cast(pl.Float64).alias("observed"),
                       pl.len().alias("n"))
                  .with_columns(gap.alias("_gap"))
                  .group_by(column)
                  .agg((pl.col("_gap").sum() / pl.col("n").sum()).alias("ece")))


def audit(scores: pl.DataFrame, outcomes: pl.DataFrame, cohort: pl.DataFrame, *,
          dates: Sequence[date] | None = None, n_bins: int = cal.BINS,
          weights: dict[str, float] | None = None) -> pl.DataFrame:
    """ECE, FNR and the relative gap for every group of every stratum, worst first."""
    dates = cal.holdout_dates(scores, outcomes) if dates is None else list(dates)
    window = scores.filter(pl.col("date").is_in(dates))

    strangers = window.join(cohort.select("veteran_id"), on="veteran_id", how="anti")
    if strangers.height:
        ids = strangers["veteran_id"].unique().to_list()
        raise ValueError(
            f"{len(ids)} scored veterans are not in the cohort, e.g. {ids[0]!r}. They have no "
            "group, so every stratum's denominator would be short by an unknown amount.")

    pairs = cal.paired(scores, outcomes, dates=dates)
    groups = strata(cohort)
    binned = cal.assign_bins(pairs, cal.bin_edges(pairs, n_bins=n_bins)).join(
        groups, on="veteran_id", how="left")
    days = veteran_days(pairs, window, weights).join(groups, on="veteran_id", how="left")

    events = int(days["needed"].sum())
    missed = int((days["needed"] & ~days["reached"]).sum())
    cohort_fnr = missed / events if events else 0.0

    frames = []
    for s in STRATA:
        counted = (days.group_by(s.name)
                       .agg(pl.len().cast(pl.Int32).alias("n"),
                            pl.col("needed").sum().cast(pl.Int32).alias("n_events"),
                            (pl.col("needed") & ~pl.col("reached")).sum()
                            .cast(pl.Int32).alias("n_missed")))
        fnr = pl.when(pl.col("n_events") > 0) \
                .then(pl.col("n_missed") / pl.col("n_events")).otherwise(0.0)
        # A stratum whose cohort FNR is 0 has every group at 0 too -- it is the pooled
        # average of them -- so a ratio of 1.0 is exact here, not a fallback.
        ratio = (pl.col("fnr") / cohort_fnr) if cohort_fnr > 0 else pl.lit(1.0)
        frames.append(
            counted.join(_ece_by_group(binned, s.name), on=s.name, how="left")
                   .rename({s.name: "group"})
                   .with_columns(pl.lit(s.name).alias("stratum"),
                                 pl.col("ece").fill_null(0.0),
                                 fnr.alias("fnr"))
                   .with_columns(ratio.alias("fnr_ratio_to_cohort"))
                   .with_columns((pl.col("fnr_ratio_to_cohort") > 1 + FNR_GAP).alias("flagged"))
                   .select(list(AUDIT_SCHEMA)))
    return (pl.concat(frames)
              .sort(pl.col("stratum").cast(STRATUM_ORDER), "fnr_ratio_to_cohort", "group",
                    descending=[False, True, False]))


def failed(tbl: pl.DataFrame) -> bool:
    """True when any group is flagged. `make report` shows the table either way."""
    return bool(tbl["flagged"].any())


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

FLAG_FILL, FLAG_INK = "#f8e3e3", "#d03b3b"          # status: critical
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID = "#fcfcfb", "#e1e0d9"


def audit_table(tbl: pl.DataFrame) -> go.Figure:
    """The audit as a table. A flagged row is marked by a word as well as a colour."""
    flag = tbl["flagged"].to_list()
    fills = [[FLAG_FILL if f else SURFACE for f in flag]] * 7
    inks = [[FLAG_INK if f else INK_2 for f in flag]] * 7
    fig = go.Figure(go.Table(
        columnwidth=[120, 150, 70, 70, 80, 90, 110],
        header={"values": ["<b>stratum</b>", "<b>group</b>", "<b>veteran-days</b>",
                           "<b>events</b>", "<b>ECE</b>", "<b>FNR</b>",
                           "<b>vs cohort</b>"],
                "fill_color": SURFACE, "align": "left",
                "font": {"color": INK, "size": 12}, "line_color": GRID},
        cells={"values": [
            tbl["stratum"].to_list(),
            [f"{g}  ⚑ flagged" if f else g for g, f in zip(tbl["group"], flag, strict=True)],
            [f"{v:,}" for v in tbl["n"]],
            [f"{v:,}" for v in tbl["n_events"]],
            [f"{v:.3f}" for v in tbl["ece"]],
            [f"{v:.3f}" for v in tbl["fnr"]],
            [f"{v:.2f}x" for v in tbl["fnr_ratio_to_cohort"]],
        ], "fill_color": fills, "align": "left", "height": 24,
            "font": {"color": inks, "size": 11}, "line_color": GRID},
    ))
    n_flagged = sum(flag)
    verdict = (f"{n_flagged} group(s) flagged: FNR more than {FNR_GAP:.0%} above the cohort"
               if n_flagged else f"no group is more than {FNR_GAP:.0%} above the cohort")
    fig.update_layout(
        title={"text": "Fairness audit: calibration and missed events by group",
               "subtitle": {"text": f"{verdict} · simulated outcomes, synthetic cohort",
                            "font": {"color": FLAG_INK if n_flagged else INK_2, "size": 13}},
               "font": {"color": INK, "size": 18}, "x": 0.02, "xanchor": "left"},
        width=820, height=max(360, 24 * tbl.height + 140),
        margin={"l": 24, "r": 24, "t": 96, "b": 24},
        paper_bgcolor=SURFACE,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": INK_2},
    )
    return fig


def write_outputs(tbl: pl.DataFrame, out_dir: Path = REPORT) -> tuple[Path, Path]:
    """report/fairness.csv (every group, flagged or not) and .html (Plotly, works offline)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, html = out_dir / "fairness.csv", out_dir / "fairness.html"
    tbl.select(list(AUDIT_SCHEMA)).write_csv(csv)
    # include_plotlyjs=True inlines plotly.js: bigger file, but it opens with the wifi off.
    audit_table(tbl).write_html(html, include_plotlyjs=True, full_html=True)
    return csv, html


def main() -> int:
    tbl = audit(schema.read("scores"), schema.read("outcomes"), schema.read("cohort"))
    csv, html = write_outputs(tbl)
    with pl.Config(tbl_rows=tbl.height + 1, tbl_width_chars=120):
        print(tbl)
    flagged = tbl.filter(pl.col("flagged"))
    if flagged.height:
        print(f"\nFAIRNESS AUDIT FAILED: {flagged.height} group(s) miss more than "
              f"{FNR_GAP:.0%} more events than the cohort. This is displayed, not suppressed.")
    else:
        print(f"\nno group exceeds the {FNR_GAP:.0%} relative FNR gap")
    print(f"wrote {csv.relative_to(schema.ROOT)} and {html.relative_to(schema.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

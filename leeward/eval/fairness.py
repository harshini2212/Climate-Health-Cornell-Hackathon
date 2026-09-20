"""Fairness audit -- who does this model miss? SPEC §11.

Every group in every stratum gets four numbers:

    ECE       how well calibrated the risks are here, against the whole window's bins
    FNR       of the veteran-days where a need actually occurred, the share where nobody
              was put in front of a human
    reach     1 - FNR: the share that *was* put in front of a human
    coverage  the share that got one of the `CALL_BUDGET` calls a day that actually exist

and the comparisons that matter, `fnr_ratio_to_cohort`, `reach_ratio_to_cohort` and
`coverage_ratio_to_cohort`. A group whose FNR ratio exceeds 1 + 20 percent is **flagged**.

**A clean audit is not the same as a fair one, and on this cohort it cannot be.** At a
pooled FNR of `f`, clearing the bar takes a group FNR of `f * 1.2`; above f = 0.833 that is
a rate greater than 1, so no group can flag however badly it is served. The held-out window
has ~16.7k veteran-days with a need and 40 calls a day to spend on them, which puts every
group's FNR between 0.93 and 0.99 -- two numbers at the ceiling cannot diverge by 20
percent. `flag_is_reachable()` says which regime a report is in, and every rendering of this
table has to repeat it. Otherwise "0 of 33 groups flagged" reads as evidence of fairness
when it is evidence of a budget.

So the flag is kept exactly where SPEC §11 put it, and `direction` is added beside it:
which side of the *same* 20 percent bar the group's **reach** falls on. Reach is FNR with
the ceiling subtracted off -- the identical measurement, on a scale where a difference is
visible. On the real cohort it separates groups by a factor of six where FNR separates them
by a tenth. A group reached *more* than the cohort gets `reached_more`, because that is a
finding too and a table full of "not flagged" hides it.

`coverage` is the one a care team can actually move: reach asks whether anybody looked,
coverage asks whether anybody phoned. The two disagree, and the disagreement is worth
reading -- the tier rule surfaces the most vulnerable on several axes, and under a 40-call
budget the allocator only follows through on some of them.

Four choices worth stating out loud, because they are the ones that decide what the audit
can see:

1. **A false negative is defined by the shipped rule, not by an eval-only threshold.** A
   veteran-day counts as reached when `leeward.decision.tiers` puts it in `act_now` or
   `find_out` -- the two tiers that cost a real person's attention. Auditing some other
   cutoff would audit a system nobody is running.
2. **Every group is binned on the whole window's edges** (`calibration.bin_edges`). Re-binning
   inside each group would make two groups' ECEs incomparable, which is the quiet way to
   make a calibration gap disappear.
3. **Only the harmed side is flagged.** A group the model misses *less* often than the
   cohort does not raise the alarm; it is labelled `reached_more` instead, and the ratios are
   in the table either way so a reader can see both directions. `n_events` and `n_called` are
   there too, so a ratio computed off a handful of events can be read as the noise it is
   rather than trusted as a verdict.
4. **Coverage is measured against the shipped allocator, not against a wish.**
   `budget_calls()` runs `leeward.decision.allocate` day by day under a K-call budget and
   nothing else -- the same call `leeward.eval.decision_quality` scores the baselines
   against. An audit given no call list reports coverage as **null**, never as zero: not
   measured and measured-as-nobody are different claims.

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
from leeward.schema import DEFAULT_CAPACITY

REPORT = schema.ROOT / "report"

#: A group whose FNR exceeds the cohort's by more than this share is flagged. The same bar,
#: pointed at `reach` instead, is what `direction` reports -- one bar, two tails.
FNR_GAP = 0.20

#: The scarce unit coverage is measured in: one care-team call, from `DEFAULT_CAPACITY`.
#: A real person's afternoon, and the reason the FNR is where it is.
CALL_BUDGET = DEFAULT_CAPACITY["call"]

#: The tiers that put a veteran in front of a human. Everything else is a miss.
REACHED_TIERS = ("act_now", "find_out")

#: `direction`: which side of the 20 percent bar a group's reach falls on. `REACHED_MORE`
#: is a result, not the absence of one, and is deliberately not spelled like a pass.
REACHED_MORE, REACHED_LESS, ON_PAR = "reached_more", "reached_less", "on_par"

UNKNOWN = "unknown"

AUDIT_SCHEMA = {"stratum": pl.Utf8, "group": pl.Utf8, "n": pl.Int32, "n_events": pl.Int32,
                "n_missed": pl.Int32, "n_called": pl.Int32, "ece": pl.Float64,
                "fnr": pl.Float64, "fnr_ratio_to_cohort": pl.Float64,
                "reach": pl.Float64, "reach_ratio_to_cohort": pl.Float64,
                "coverage": pl.Float64, "coverage_ratio_to_cohort": pl.Float64,
                "direction": pl.Utf8, "flagged": pl.Boolean}

#: The columns of `FairnessRow` -- what `report.json` carries, and so what the screen can
#: show. `fnr` stays in it: it is the honest denominator, and swapping it for the friendlier
#: number rather than showing both would be the suppression this audit exists to prevent.
#: `n_events` and `n_called` travel too, because several coverage ratios here are computed
#: off single-figure counts and a ratio on screen without its denominator invites a reader
#: to trust noise. `n_missed` is recoverable from `n_events` and `fnr`, and stays in the CSV.
REPORT_COLUMNS = ["stratum", "group", "n", "n_events", "n_called", "ece",
                  "fnr", "fnr_ratio_to_cohort", "reach", "reach_ratio_to_cohort",
                  "coverage", "coverage_ratio_to_cohort", "direction", "flagged"]


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


def budget_calls(scores: pl.DataFrame, cohort: pl.DataFrame, *, k: int = CALL_BUDGET,
                 dates: Sequence[date] | None = None) -> pl.DataFrame:
    """The veteran-days that got one of `k` care-team calls, day by day.

    The allocator is given K calls and no other capacity -- the same budget
    `decision_quality` hands every strategy -- so this is the list the care team would
    actually have worked, not the list the tier rule would have liked them to. It is
    deterministic: `allocate` breaks ties on (value, veteran, action), so the same window
    gives the same list in rehearsal and on stage.

    Fewer than K rows on a day is normal and not an error: the allocator can spend a call
    bucket on a veteran it has already chosen, and on a quiet day it may find nothing worth
    a call at all.
    """
    from leeward.decision.allocate import allocate
    from leeward.eval import decision_quality as dq

    days = sorted(scores["date"].unique().to_list()) if dates is None else list(dates)
    picked = [dq.leeward_actions(allocate, scores.filter(pl.col("date") == day), cohort, k)
              .select("veteran_id", "date").unique()
              for day in days]
    return (pl.concat(picked) if picked else
            pl.DataFrame(schema={"veteran_id": pl.Utf8, "date": pl.Date}))


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

def _mark_calls(days: pl.DataFrame, calls: pl.DataFrame | None) -> pl.DataFrame:
    """Add `called` to the veteran-day panel: did this day get one of the scarce calls?"""
    if calls is None:
        return days
    flags = calls.select("veteran_id", "date").unique().with_columns(
        pl.lit(True).alias("called"))
    return (days.join(flags, on=["veteran_id", "date"], how="left")
                .with_columns(pl.col("called").fill_null(False)))


def _called_agg(calls: pl.DataFrame | None) -> list[pl.Expr]:
    """`n_called` per group, or an explicit null column when no call list was given."""
    if calls is None:
        return [pl.lit(None, dtype=pl.Int32).alias("n_called")]
    return [(pl.col("needed") & pl.col("called")).sum().cast(pl.Int32).alias("n_called")]


def _ratio(column: str, pooled: float | None, scored: pl.Expr) -> pl.Expr:
    """`column` over its pooled value, with the two degenerate cases pinned to 1.0.

    A pooled rate of 0 means every group is 0 too -- it is their weighted average -- so 1.0
    is exact there rather than a fallback. A group with no events has no rate to compare,
    and sits at 1.0 so it is never mistaken for the best or the worst group in the table.
    A pooled value of None means the quantity was not measured, and stays null.
    """
    if pooled is None:
        return pl.lit(None, dtype=pl.Float64)
    ratio = (pl.col(column) / pooled) if pooled > 0 else pl.lit(1.0)
    return pl.when(scored).then(ratio).otherwise(1.0)


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
          weights: dict[str, float] | None = None,
          calls: pl.DataFrame | None = None) -> pl.DataFrame:
    """ECE, FNR, reach and coverage for every group of every stratum, worst first.

    `calls` is a (veteran_id, date) frame of veteran-days that got a scarce call, usually
    from `budget_calls()`. Leave it out and the coverage columns are null -- an audit that
    was not given a call list has not measured coverage, and saying 0 would be a claim it
    has no evidence for.
    """
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
    days = _mark_calls(days, calls)

    events = int(days["needed"].sum())
    missed = int((days["needed"] & ~days["reached"]).sum())
    pooled_fnr = missed / events if events else 0.0
    pooled_reach = 1 - pooled_fnr
    called = None if calls is None else int((days["needed"] & days["called"]).sum())
    pooled_cov = None if called is None else (called / events if events else 0.0)

    # 0 of 0 events is neither a success nor a failure, so a group with an empty denominator
    # is parked at the cohort's own ratio rather than being scored on a rate it does not
    # have. `n_events` is in the table so that shows up as the absence of evidence it is.
    scored = pl.col("n_events") > 0

    frames = []
    for s in STRATA:
        counted = (days.group_by(s.name)
                       .agg(pl.len().cast(pl.Int32).alias("n"),
                            pl.col("needed").sum().cast(pl.Int32).alias("n_events"),
                            (pl.col("needed") & ~pl.col("reached")).sum()
                            .cast(pl.Int32).alias("n_missed"),
                            *_called_agg(calls)))
        fnr = pl.when(scored).then(pl.col("n_missed") / pl.col("n_events")).otherwise(0.0)
        reach = pl.when(scored).then(1 - pl.col("fnr")).otherwise(0.0)
        cov = pl.when(scored).then(pl.col("n_called") / pl.col("n_events")).otherwise(0.0)
        direction = (pl.when(pl.col("reach_ratio_to_cohort") > 1 + FNR_GAP)
                       .then(pl.lit(REACHED_MORE))
                       .when(pl.col("reach_ratio_to_cohort") < 1 - FNR_GAP)
                       .then(pl.lit(REACHED_LESS))
                       .otherwise(pl.lit(ON_PAR)))
        frames.append(
            counted.join(_ece_by_group(binned, s.name), on=s.name, how="left")
                   .rename({s.name: "group"})
                   .with_columns(pl.lit(s.name).alias("stratum"),
                                 pl.col("ece").fill_null(0.0),
                                 fnr.alias("fnr"),
                                 cov.alias("coverage") if calls is not None else
                                 pl.lit(None, dtype=pl.Float64).alias("coverage"))
                   .with_columns(reach.alias("reach"),
                                 _ratio("fnr", pooled_fnr, scored).alias("fnr_ratio_to_cohort"),
                                 _ratio("coverage", pooled_cov, scored)
                                 .alias("coverage_ratio_to_cohort"))
                   .with_columns(_ratio("reach", pooled_reach, scored)
                                 .alias("reach_ratio_to_cohort"))
                   .with_columns(direction.alias("direction"),
                                 (pl.col("fnr_ratio_to_cohort") > 1 + FNR_GAP).alias("flagged"))
                   .select(list(AUDIT_SCHEMA)))
    return (pl.concat(frames)
              .sort(pl.col("stratum").cast(STRATUM_ORDER), "fnr_ratio_to_cohort", "group",
                    descending=[False, True, False]))


def failed(tbl: pl.DataFrame) -> bool:
    """True when any group is flagged. `make report` shows the table either way."""
    return bool(tbl["flagged"].any())


def _pooled(tbl: pl.DataFrame, numerator: str) -> float | None:
    """Recover a pooled rate from the table. Every stratum covers the same veteran-days, so
    summing across all of them scales numerator and denominator alike and the rate holds."""
    events = tbl["n_events"].sum()
    got = tbl[numerator].sum()
    if got is None:
        return None
    return got / events if events else 0.0


def cohort_fnr(tbl: pl.DataFrame) -> float:
    """The pooled FNR every `fnr_ratio_to_cohort` in the table was taken against."""
    return _pooled(tbl, "n_missed") or 0.0


def cohort_coverage(tbl: pl.DataFrame) -> float | None:
    """The pooled coverage, or None when the audit was run without a call list."""
    if tbl["n_called"].null_count() == tbl.height:
        return None
    return _pooled(tbl, "n_called")


def flag_is_reachable(pooled_fnr: float, gap: float = FNR_GAP) -> bool:
    """Can *any* group clear the relative FNR bar at this pooled rate?

    Clearing it takes a group FNR of `pooled_fnr * (1 + gap)`, and an FNR above 1 does not
    exist. Above a pooled 0.833 the answer is therefore no, whatever the model does and
    however unevenly it does it -- which is the single most important thing to say next to a
    report where nothing is flagged.
    """
    return pooled_fnr * (1 + gap) <= 1.0


def ceiling_note(tbl: pl.DataFrame) -> str:
    """One line saying why the flags read the way they do. Rendered, never omitted."""
    f, cov = cohort_fnr(tbl), cohort_coverage(tbl)
    events = tbl["n_events"].sum() // max(len(STRATA), 1)
    budget = "" if cov is None else (
        f" Only {CALL_BUDGET} calls a day exist to spend on them, and they reached "
        f"{cov:.1%} of those events.")
    if flag_is_reachable(f):
        return (f"Pooled FNR {f:.3f} over {events:,} veteran-days with a need. A group would "
                f"have to miss {f * (1 + FNR_GAP):.3f} of its events to be flagged.{budget}")
    return (f"Pooled FNR is {f:.3f} over {events:,} veteran-days with a need, so flagging a "
            f"group would take an FNR of {f * (1 + FNR_GAP):.3f} -- impossible, and no group "
            f"can be flagged however it is served.{budget} Read the reach and coverage "
            f"ratios, not the level.")


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

FLAG_FILL, FLAG_INK = "#f8e3e3", "#d03b3b"          # status: critical
MORE_FILL, MORE_INK = "#e6f0e8", "#2f6b45"          # status: reached more than the cohort
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID = "#fcfcfb", "#e1e0d9"

#: The header of every rendered column, in order, and how to read each cell out of a row.
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("stratum", "stratum"), ("group", "group"), ("events", "n_events"), ("ECE", "ece"),
    ("FNR", "fnr"), ("reach", "reach"), ("reach vs cohort", "reach_ratio_to_cohort"),
    ("coverage", "coverage"), ("cov. vs cohort", "coverage_ratio_to_cohort"),
    ("direction", "direction"),
)


def _rate(v: float | None) -> str:
    """A rate, or the fact that it was never measured. Never a silent 0."""
    return "not measured" if v is None else f"{v:.3f}"


def _times(v: float | None) -> str:
    return "—" if v is None else f"{v:.2f}x"


def _cells(tbl: pl.DataFrame) -> list[list[str]]:
    flag = tbl["flagged"].to_list()
    return [
        tbl["stratum"].to_list(),
        [f"{g}  ⚑ flagged" if f else g for g, f in zip(tbl["group"], flag, strict=True)],
        [f"{v:,}" for v in tbl["n_events"]],
        [f"{v:.3f}" for v in tbl["ece"]],
        [f"{v:.3f}" for v in tbl["fnr"]],
        [f"{v:.3f}" for v in tbl["reach"]],
        [_times(v) for v in tbl["reach_ratio_to_cohort"]],
        [_rate(v) for v in tbl["coverage"]],
        [_times(v) for v in tbl["coverage_ratio_to_cohort"]],
        [d.replace("_", " ") for d in tbl["direction"]],
    ]


def audit_table(tbl: pl.DataFrame) -> go.Figure:
    """The audit as a table.

    A flagged row is marked by a word as well as a colour, and so is a group reached *more*
    than the cohort -- on this data that is the only row type the FNR bar can ever produce,
    and leaving it unmarked would render the audit's one finding as blank space.
    """
    flag, direction = tbl["flagged"].to_list(), tbl["direction"].to_list()
    fill = [FLAG_FILL if f else MORE_FILL if d == REACHED_MORE else SURFACE
            for f, d in zip(flag, direction, strict=True)]
    ink = [FLAG_INK if f else MORE_INK if d == REACHED_MORE else INK_2
           for f, d in zip(flag, direction, strict=True)]
    n_col = len(_COLUMNS)
    fig = go.Figure(go.Table(
        columnwidth=[110, 140, 70, 65, 65, 65, 105, 90, 105, 100],
        header={"values": [f"<b>{h}</b>" for h, _ in _COLUMNS],
                "fill_color": SURFACE, "align": "left",
                "font": {"color": INK, "size": 12}, "line_color": GRID},
        cells={"values": _cells(tbl), "fill_color": [fill] * n_col, "align": "left",
               "height": 24, "font": {"color": [ink] * n_col, "size": 11},
               "line_color": GRID},
    ))
    n_flagged, n_more = sum(flag), direction.count(REACHED_MORE)
    verdict = (f"{n_flagged} group(s) flagged: FNR more than {FNR_GAP:.0%} above the cohort"
               if n_flagged else f"no group is more than {FNR_GAP:.0%} above the cohort")
    verdict += f" · {n_more} group(s) reached more than the cohort"
    fig.update_layout(
        title={"text": "Fairness audit: calibration, missed events and coverage by group",
               "subtitle": {"text": f"{verdict}<br>{ceiling_note(tbl)}"
                                    "<br>simulated outcomes, synthetic cohort",
                            "font": {"color": FLAG_INK if n_flagged else INK_2, "size": 12}},
               "font": {"color": INK, "size": 18}, "x": 0.02, "xanchor": "left"},
        width=1060, height=max(380, 24 * tbl.height + 190),
        margin={"l": 24, "r": 24, "t": 148, "b": 24},
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


def console_verdict(tbl: pl.DataFrame) -> str:
    """The audit's verdict in words, including the half a flag count cannot carry."""
    flagged = tbl.filter(pl.col("flagged"))
    more = tbl.filter(pl.col("direction") == REACHED_MORE)
    lines = []
    if flagged.height:
        lines.append(f"FAIRNESS AUDIT FAILED: {flagged.height} group(s) miss more than "
                     f"{FNR_GAP:.0%} more events than the cohort. This is displayed, not "
                     "suppressed.")
    else:
        lines.append(f"no group exceeds the {FNR_GAP:.0%} relative FNR gap")
    lines.append(ceiling_note(tbl))
    if more.height:
        top = more.sort("reach_ratio_to_cohort", descending=True).head(4)
        named = ", ".join(f"{r['stratum']}:{r['group']} {r['reach_ratio_to_cohort']:.2f}x"
                          for r in top.iter_rows(named=True))
        lines.append(f"{more.height} group(s) reached MORE than the cohort -- a result, not "
                     f"an absence of one: {named}")
    return "\n".join(lines)


def main() -> int:
    scores, outcomes = schema.read("scores"), schema.read("outcomes")
    cohort = schema.read("cohort")
    dates = cal.holdout_dates(scores, outcomes)
    calls = budget_calls(scores, cohort, dates=dates)
    tbl = audit(scores, outcomes, cohort, dates=dates, calls=calls)
    csv, html = write_outputs(tbl)
    with pl.Config(tbl_rows=tbl.height + 1, tbl_width_chars=200):
        print(tbl.select(REPORT_COLUMNS))
    print()
    print(console_verdict(tbl))
    print(f"wrote {csv.relative_to(schema.ROOT)} and {html.relative_to(schema.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

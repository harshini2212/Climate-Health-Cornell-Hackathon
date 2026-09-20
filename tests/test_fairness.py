"""fairness.py -- who does this model miss? Written before the module.

The audit asks two questions of every group: is the model as well calibrated here as it is
everywhere (ECE), and does it miss as few real events here as it does everywhere (false
negative rate). A group whose relative FNR gap exceeds 20 percent is `flagged`.

These tests exist mostly to make the audit hard to soften. A group is never dropped for
being small, an unknown race is its own group rather than a deleted row, the flag fires on
the ratio and nothing else, and `failed()` is true the moment one row is flagged. The
guardrail in `test_guardrails.py` covers the other half: no swallowed exceptions.

The flag is only half of what the audit has to say, because on the real cohort it is a bar
nothing can clear: at a pooled FNR of 0.96, exceeding it by 20 percent would take an FNR of
1.15. So three more quantities are tested here, and `reach` and `direction` are the ones a
reader is meant to look at:

    reach      1 - FNR, the share of a group's events that reached a human at all
    coverage   the share that got one of the 40 calls a day that actually exist
    direction  which side of the same 20 percent bar the group's *reach* falls on

`direction` is judged on reach rather than on FNR on purpose. Both are the same number, but
only one of them is off the ceiling, and a bar that cannot be crossed cannot report a
finding. A group reached *more* than the cohort is a result -- `reached_more` says so in the
output, without touching the flag.
"""

from __future__ import annotations

import re
from datetime import date

import polars as pl
import pytest

from leeward.api.schemas import FairnessRow
from leeward.eval import calibration as cal
from leeward.eval import fairness as fair
from leeward.schema import NEEDS
from tables import table

D0 = date(2026, 7, 1)

#: A veteran-day that reaches the care team: act_now needs p >= 0.25 on a need with w >= 4
#: and an epistemic share under 0.4 (SPEC §7.5).
LOUD, QUIET = 0.9, 0.01


def _cohort(rows: list[dict]) -> pl.DataFrame:
    """The nine cohort columns the audit reads, and nothing else."""
    base = {"borough": "Bronx", "hvi": 3, "evac_zone": 0, "income_band": "mid",
            "caregiver": "none", "n_active_meds": 2, "race": "White",
            "ethnicity": "Non-Hispanic"}
    return pl.DataFrame([base | r for r in rows],
                        schema={"veteran_id": pl.Utf8, "borough": pl.Utf8, "hvi": pl.Int32,
                                "evac_zone": pl.Int32, "income_band": pl.Utf8,
                                "caregiver": pl.Utf8, "n_active_meds": pl.Int32,
                                "race": pl.Utf8, "ethnicity": pl.Utf8})


def _world(spec: list[tuple[str, bool, bool]], day: date = D0
           ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(scores, outcomes) for `(veteran_id, reached, had_an_event)` triples.

    Reached veterans are scored loud on treatment_gap so the shipped tier rule puts them in
    front of a human; the rest are scored quiet on every need.
    """
    srows, orows = [], []
    for vid, reached, event in spec:
        for need in NEEDS:
            p = LOUD if (reached and need == "treatment_gap") else QUIET
            srows.append({"veteran_id": vid, "date": day, "need": need,
                          "p_mean": p, "p_epistemic_share": 0.1})
            orows.append({"veteran_id": vid, "date": day, "need": need,
                          "y": int(event and need == "treatment_gap")})
    scores = pl.DataFrame(srows, schema={"veteran_id": pl.Utf8, "date": pl.Date,
                                         "need": pl.Utf8, "p_mean": pl.Float64,
                                         "p_epistemic_share": pl.Float64})
    outcomes = pl.DataFrame(orows, schema={"veteran_id": pl.Utf8, "date": pl.Date,
                                           "need": pl.Utf8, "y": pl.Int32})
    return scores, outcomes


def _boroughs(bronx: list[tuple[bool, bool]], queens: list[tuple[bool, bool]]):
    spec, rows = [], []
    for borough, people in (("Bronx", bronx), ("Queens", queens)):
        for i, (reached, event) in enumerate(people):
            vid = f"{borough[:2]}{i}"
            spec.append((vid, reached, event))
            rows.append({"veteran_id": vid, "borough": borough})
    scores, outcomes = _world(spec)
    return scores, outcomes, _cohort(rows)


def _borough_rows(tbl: pl.DataFrame) -> dict[str, dict]:
    return {r["group"]: r for r in tbl.filter(pl.col("stratum") == "borough").to_dicts()}


# --------------------------------------------------------------------------- #
# The false negative rate
# --------------------------------------------------------------------------- #

def test_fnr_is_events_the_care_team_never_heard_about_over_events() -> None:
    """Bronx: five events, one veteran reached -> 4/5. Queens: five events, four reached
    -> 1/5. Nobody without an event is in either denominator."""
    scores, outcomes, cohort = _boroughs(
        bronx=[(True, True)] + [(False, True)] * 4 + [(False, False)] * 3,
        queens=[(True, True)] * 4 + [(False, True)] + [(False, False)] * 3)
    rows = _borough_rows(fair.audit(scores, outcomes, cohort))
    assert rows["Bronx"]["fnr"] == pytest.approx(0.8)
    assert rows["Queens"]["fnr"] == pytest.approx(0.2)
    assert rows["Bronx"]["n_events"] == rows["Queens"]["n_events"] == 5
    assert rows["Bronx"]["n_missed"] == 4


def test_n_is_the_group_size_in_veteran_days_not_the_event_count() -> None:
    scores, outcomes, cohort = _boroughs(bronx=[(True, True)] * 3,
                                         queens=[(False, True)] * 7)
    rows = _borough_rows(fair.audit(scores, outcomes, cohort))
    assert (rows["Bronx"]["n"], rows["Queens"]["n"]) == (3, 7)


def test_a_veteran_day_counts_once_however_many_needs_occurred() -> None:
    """Reaching someone is a phone call, not five phone calls. The denominator is people."""
    scores, outcomes = _world([("A", False, True)])
    outcomes = outcomes.with_columns(pl.lit(1, dtype=pl.Int32).alias("y"))   # all five needs
    rows = _borough_rows(fair.audit(scores, outcomes, _cohort([{"veteran_id": "A"}])))
    assert rows["Bronx"]["n_events"] == 1
    assert rows["Bronx"]["fnr"] == pytest.approx(1.0)


def test_a_group_with_no_events_reports_zero_and_says_why() -> None:
    """0 of 0 missed is not a pass. `n_events` is in the table so nobody reads it as one.

    It is not a *success* either: a group with nothing in the denominator has a reach of 0
    over 0, so it sits at the cohort's own ratio and `on_par`, never `reached_more`.
    """
    scores, outcomes, cohort = _boroughs(bronx=[(False, True)] * 4,
                                         queens=[(False, False)] * 4)
    rows = _borough_rows(fair.audit(scores, outcomes, cohort))
    assert rows["Queens"]["n_events"] == 0
    assert rows["Queens"]["fnr"] == 0.0
    assert rows["Queens"]["flagged"] is False
    assert rows["Queens"]["reach_ratio_to_cohort"] == 1.0
    assert rows["Queens"]["direction"] == fair.ON_PAR


# --------------------------------------------------------------------------- #
# The flag
# --------------------------------------------------------------------------- #

def test_a_group_missing_60_percent_more_events_than_the_cohort_is_flagged() -> None:
    scores, outcomes, cohort = _boroughs(
        bronx=[(True, True)] + [(False, True)] * 4,
        queens=[(True, True)] * 4 + [(False, True)])
    rows = _borough_rows(fair.audit(scores, outcomes, cohort))
    assert rows["Bronx"]["fnr_ratio_to_cohort"] == pytest.approx(0.8 / 0.5)
    assert rows["Bronx"]["flagged"] is True
    assert rows["Queens"]["flagged"] is False


def test_a_gap_of_exactly_20_percent_is_not_yet_a_flag() -> None:
    """Cohort FNR 0.5, Bronx 0.6: a ratio of exactly 1.2. The bar is *exceeds* 20 percent."""
    scores, outcomes, cohort = _boroughs(bronx=[(False, True)] * 6 + [(True, True)] * 4,
                                         queens=[(False, True)] * 4 + [(True, True)] * 6)
    rows = _borough_rows(fair.audit(scores, outcomes, cohort))
    assert rows["Bronx"]["fnr_ratio_to_cohort"] == 1.2
    assert rows["Bronx"]["flagged"] is False


def test_the_flag_is_the_ratio_and_nothing_else() -> None:
    tbl = fair.audit(table("scores"), table("outcomes"), table("cohort"))
    assert tbl["flagged"].to_list() == [r > 1 + fair.FNR_GAP
                                        for r in tbl["fnr_ratio_to_cohort"]]


def test_a_group_the_model_serves_better_than_average_is_not_flagged() -> None:
    """The audit looks for harm. Missing *fewer* events than the cohort is not a finding."""
    tbl = fair.audit(table("scores"), table("outcomes"), table("cohort"))
    # The cohort FNR is the weighted average of its groups, so some group is always at or
    # below it: this filter cannot silently come back empty and pass on nothing.
    better = tbl.filter(pl.col("fnr_ratio_to_cohort") <= 1.0)
    assert better.height, "the pooled rate is a weighted average; something must be under it"
    assert not better["flagged"].any()


def test_failed_is_true_the_moment_one_row_is_flagged() -> None:
    scores, outcomes, cohort = _boroughs(bronx=[(False, True)] * 5,
                                         queens=[(True, True)] * 5)
    assert fair.failed(fair.audit(scores, outcomes, cohort)) is True
    even = _boroughs(bronx=[(True, True)] * 5, queens=[(True, True)] * 5)
    assert fair.failed(fair.audit(*even)) is False


# --------------------------------------------------------------------------- #
# Direction: which way a group misses, on a bar that can actually be crossed
# --------------------------------------------------------------------------- #

def test_a_group_deliberately_under_reached_is_flagged_and_pointed_at() -> None:
    """Bronx reaches nobody, Queens reaches everybody, and the audit says which is which."""
    scores, outcomes, cohort = _boroughs(bronx=[(False, True)] * 5,
                                         queens=[(True, True)] * 5)
    rows = _borough_rows(fair.audit(scores, outcomes, cohort))
    assert rows["Bronx"]["flagged"] is True
    assert rows["Bronx"]["direction"] == fair.REACHED_LESS
    assert rows["Queens"]["flagged"] is False
    assert rows["Queens"]["direction"] == fair.REACHED_MORE


def test_swapping_the_two_groups_moves_the_flag_to_the_other_one() -> None:
    """The same cohort with the labels exchanged. Nothing about the audit is attached to
    the name 'Bronx'; if it were, the finding would be an artefact of the fixture."""
    scores, outcomes, cohort = _boroughs(bronx=[(True, True)] * 5,
                                         queens=[(False, True)] * 5)
    rows = _borough_rows(fair.audit(scores, outcomes, cohort))
    assert rows["Queens"]["flagged"] is True
    assert rows["Queens"]["direction"] == fair.REACHED_LESS
    assert rows["Bronx"]["flagged"] is False
    assert rows["Bronx"]["direction"] == fair.REACHED_MORE


def test_a_group_reached_more_is_not_flagged_but_is_visible_in_the_output() -> None:
    """The production case in miniature, and the reason `direction` exists.

    Bronx misses 9 of 10, Queens 7 of 10, so the cohort misses 16 of 20. Both FNR ratios
    (1.125 and 0.875) sit inside the 20 percent band and neither flags -- yet Queens reaches
    three times as many of its events as Bronx does. On reach the same bar separates them
    cleanly, which is the whole point: the audit still has something to report.
    """
    scores, outcomes, cohort = _boroughs(
        bronx=[(True, True)] + [(False, True)] * 9,
        queens=[(True, True)] * 3 + [(False, True)] * 7)
    rows = _borough_rows(fair.audit(scores, outcomes, cohort))

    assert rows["Bronx"]["fnr_ratio_to_cohort"] == pytest.approx(0.9 / 0.8)
    assert rows["Queens"]["fnr_ratio_to_cohort"] == pytest.approx(0.7 / 0.8)
    assert not any(r["flagged"] for r in rows.values()), "neither group clears the FNR bar"

    assert rows["Queens"]["reach"] == pytest.approx(0.3)
    assert rows["Queens"]["reach_ratio_to_cohort"] == pytest.approx(0.3 / 0.2)
    assert rows["Queens"]["direction"] == fair.REACHED_MORE
    assert rows["Bronx"]["direction"] == fair.REACHED_LESS


def test_reach_is_the_complement_of_fnr_and_nothing_cleverer() -> None:
    tbl = fair.audit(table("scores"), table("outcomes"), table("cohort"))
    for row in tbl.filter(pl.col("n_events") > 0).to_dicts():
        assert row["reach"] == pytest.approx(1 - row["fnr"])


def test_direction_is_the_reach_ratio_against_the_same_20_percent_bar() -> None:
    """One bar, two tails. A reader who has understood the flag has understood this."""
    tbl = fair.audit(table("scores"), table("outcomes"), table("cohort"))
    for row in tbl.to_dicts():
        r = row["reach_ratio_to_cohort"]
        want = (fair.REACHED_MORE if r > 1 + fair.FNR_GAP else
                fair.REACHED_LESS if r < 1 - fair.FNR_GAP else fair.ON_PAR)
        assert row["direction"] == want


def test_a_flag_is_unreachable_once_the_cohort_misses_almost_everything() -> None:
    """The honest reading of a clean audit, stated as arithmetic rather than as a hope.

    At a pooled FNR of f, clearing the bar takes a group FNR of f x 1.2, which above
    f = 0.833 is a rate greater than 1 and so cannot happen. `flag_is_reachable` says which
    regime the report is in, and the screen has to repeat it -- otherwise "0 of 33 flagged"
    reads as evidence of fairness when it is evidence of a budget.
    """
    assert fair.flag_is_reachable(0.5) is True
    assert fair.flag_is_reachable(1 / (1 + fair.FNR_GAP)) is True     # exactly 1.00
    assert fair.flag_is_reachable(0.96) is False

    tbl = fair.audit(table("scores"), table("outcomes"), table("cohort"))
    f = fair.cohort_fnr(tbl)
    # Every stratum partitions the same veteran-days, so pooling across all of them at once
    # multiplies numerator and denominator alike and lands on the same rate.
    assert f == pytest.approx(tbl["n_missed"].sum() / tbl["n_events"].sum())
    if not fair.flag_is_reachable(f):
        assert not tbl["flagged"].any(), "no group can exceed a bar above 1.0"


# --------------------------------------------------------------------------- #
# Coverage: of the events that happened, which got one of the day's few calls
# --------------------------------------------------------------------------- #

def _calls(*pairs: tuple[str, date]) -> pl.DataFrame:
    return pl.DataFrame({"veteran_id": [p[0] for p in pairs],
                         "date": [p[1] for p in pairs]},
                        schema={"veteran_id": pl.Utf8, "date": pl.Date})


def test_coverage_is_the_share_of_a_groups_events_that_got_a_call() -> None:
    """Reach asks whether anyone looked. Coverage asks whether anyone phoned."""
    scores, outcomes, cohort = _boroughs(bronx=[(True, True)] * 4,
                                         queens=[(True, True)] * 4)
    tbl = fair.audit(scores, outcomes, cohort,
                     calls=_calls(("Br0", D0), ("Br1", D0), ("Qu0", D0)))
    rows = _borough_rows(tbl)
    assert (rows["Bronx"]["n_called"], rows["Bronx"]["coverage"]) == (2, pytest.approx(0.5))
    assert (rows["Queens"]["n_called"], rows["Queens"]["coverage"]) == (1, pytest.approx(0.25))
    # Every one of the eight was reached, so reach cannot tell these two groups apart at all.
    assert rows["Bronx"]["reach"] == rows["Queens"]["reach"] == 1.0
    assert rows["Bronx"]["coverage_ratio_to_cohort"] == pytest.approx(0.5 / (3 / 8))


def test_a_call_to_someone_with_no_event_that_day_is_not_coverage() -> None:
    """Coverage is measured against events, not against effort."""
    scores, outcomes, cohort = _boroughs(bronx=[(True, True), (True, False)],
                                         queens=[(True, True)])
    rows = _borough_rows(fair.audit(scores, outcomes, cohort,
                                    calls=_calls(("Br1", D0))))
    assert rows["Bronx"]["n_called"] == 0
    assert rows["Bronx"]["coverage"] == 0.0


def test_coverage_is_null_when_no_budget_was_spent_never_zero() -> None:
    """An audit run without a call list has not measured coverage. Zero would be a claim."""
    scores, outcomes, cohort = _boroughs(bronx=[(True, True)] * 3,
                                         queens=[(False, True)] * 3)
    tbl = fair.audit(scores, outcomes, cohort)
    assert tbl["coverage"].null_count() == tbl.height
    assert tbl["n_called"].null_count() == tbl.height
    assert tbl["coverage_ratio_to_cohort"].null_count() == tbl.height
    assert fair.cohort_coverage(tbl) is None


def test_budget_calls_never_spends_more_than_the_budget() -> None:
    """The allocator is asked for K calls a day and gets no other capacity at all."""
    scores, cohort = table("scores"), table("cohort")
    dates = cal.holdout_dates(scores, table("outcomes"), n_days=2)
    calls = fair.budget_calls(scores, cohort, k=5, dates=dates)
    per_day = calls.group_by("date").agg(pl.len().alias("n"))
    assert set(calls.columns) == {"veteran_id", "date"}
    assert per_day["n"].max() <= 5
    assert sorted(per_day["date"].to_list()) == sorted(dates)
    assert calls.is_unique().all(), "a veteran-day is called once or not at all"


def test_budget_calls_is_seeded_and_gives_the_same_list_twice() -> None:
    """The same click has to produce the same number in rehearsal and on stage."""
    scores, cohort, outcomes = table("scores"), table("cohort"), table("outcomes")
    dates = cal.holdout_dates(scores, outcomes, n_days=2)
    first = fair.budget_calls(scores, cohort, k=5, dates=dates).sort("veteran_id", "date")
    second = fair.budget_calls(scores, cohort, k=5, dates=dates).sort("veteran_id", "date")
    assert first.equals(second)


def test_coverage_cannot_exceed_reach_because_a_call_is_a_way_of_reaching() -> None:
    """Whoever got a call was, by definition, put in front of a human."""
    scores, outcomes, cohort = table("scores"), table("outcomes"), table("cohort")
    dates = cal.holdout_dates(scores, outcomes, n_days=3)
    calls = fair.budget_calls(scores, cohort, k=fair.CALL_BUDGET, dates=dates)
    tbl = fair.audit(scores, outcomes, cohort, dates=dates, calls=calls)
    over = tbl.filter(pl.col("coverage") > pl.col("reach") + 1e-9)
    assert not over.height, f"coverage above reach in {over.height} group(s):\n{over}"


# --------------------------------------------------------------------------- #
# Calibration, per group, on the cohort's own bins
# --------------------------------------------------------------------------- #

def test_group_ece_uses_the_whole_windows_bin_edges() -> None:
    """Re-binning inside each group would make a calibration gap between two groups
    incomparable: the bins would not be the same bins."""
    scores, outcomes, cohort = table("scores"), table("outcomes"), table("cohort")
    days = cal.holdout_dates(scores, outcomes, n_days=3)
    everyone = cal.paired(scores, outcomes, dates=days)
    edges = cal.bin_edges(everyone)
    tbl = fair.audit(scores, outcomes, cohort, dates=days)

    bronx = cohort.filter(pl.col("borough") == "Bronx")["veteran_id"].to_list()
    want = cal.ece_overall(cal.reliability(
        everyone.filter(pl.col("veteran_id").is_in(bronx)), edges=edges))
    assert _borough_rows(tbl)["Bronx"]["ece"] == pytest.approx(want)


# --------------------------------------------------------------------------- #
# Nobody gets dropped
# --------------------------------------------------------------------------- #

def test_every_stratum_in_the_spec_is_audited() -> None:
    tbl = fair.audit(table("scores"), table("outcomes"), table("cohort"))
    assert set(tbl["stratum"].unique().to_list()) == {s.name for s in fair.STRATA}
    assert {"borough", "hvi_band", "evac_zone", "income_band", "caregiver",
            "medication_burden", "race", "ethnicity"} == {s.name for s in fair.STRATA}


def test_every_group_present_in_the_cohort_gets_a_row() -> None:
    cohort = table("cohort")
    tbl = fair.audit(table("scores"), table("outcomes"), cohort)
    got = set(tbl.filter(pl.col("stratum") == "borough")["group"].to_list())
    assert got == set(cohort["borough"].unique().to_list())
    assert tbl["n"].min() > 0


def test_an_unknown_race_is_its_own_group_not_a_deleted_row() -> None:
    scores, outcomes = _world([("A", True, True), ("B", False, True)])
    cohort = _cohort([{"veteran_id": "A", "race": None},
                      {"veteran_id": "B", "race": "Black"}])
    tbl = fair.audit(scores, outcomes, cohort)
    race = tbl.filter(pl.col("stratum") == "race")
    assert set(race["group"].to_list()) == {fair.UNKNOWN, "Black"}
    assert race["n"].sum() == 2


def test_the_bands_are_the_ones_the_spec_names() -> None:
    cohort = _cohort([
        {"veteran_id": "a", "hvi": 5, "evac_zone": 0, "n_active_meds": 4},
        {"veteran_id": "b", "hvi": 1, "evac_zone": 3, "n_active_meds": 5},
        {"veteran_id": "c", "hvi": 1, "evac_zone": 6, "n_active_meds": 10},
    ])
    banded = fair.strata(cohort)
    assert banded["hvi_band"].to_list() == ["HVI 5", "HVI 1", "HVI 1"]
    assert banded["evac_zone"].to_list() == ["no zone", "zone 3", "zone 6"]
    assert banded["medication_burden"].to_list() == ["0-4 meds", "5-9 meds", "10+ meds"]


def test_a_scored_veteran_missing_from_the_cohort_is_an_error() -> None:
    """An unjoined veteran would silently leave a whole group's denominator short."""
    scores, outcomes = _world([("A", True, True), ("stranger", False, True)])
    with pytest.raises(ValueError, match="stranger"):
        fair.audit(scores, outcomes, _cohort([{"veteran_id": "A"}]))


# --------------------------------------------------------------------------- #
# What comes out
# --------------------------------------------------------------------------- #

def test_audit_rows_fit_the_report_contract() -> None:
    tbl = fair.audit(table("scores"), table("outcomes"), table("cohort"))
    for row in tbl.select(fair.REPORT_COLUMNS).to_dicts():
        FairnessRow(**row)


def test_the_report_carries_the_direction_and_keeps_the_raw_rate() -> None:
    """What `report.json` shows is what the screen can show. `fnr` stays in it: it is the
    honest denominator, and dropping it for the friendlier number would be suppression."""
    assert {"fnr", "fnr_ratio_to_cohort", "reach", "reach_ratio_to_cohort", "direction",
            "coverage", "coverage_ratio_to_cohort", "flagged"} <= set(fair.REPORT_COLUMNS)
    assert set(fair.REPORT_COLUMNS) <= set(FairnessRow.model_fields)


def test_outputs_are_a_tidy_csv_and_an_offline_plotly_table(tmp_path) -> None:
    tbl = fair.audit(table("scores"), table("outcomes"), table("cohort"))
    csv, html = fair.write_outputs(tbl, tmp_path)
    back = pl.read_csv(csv)
    assert back.columns == list(fair.AUDIT_SCHEMA)
    assert back.height == tbl.height
    page = html.read_text(encoding="utf-8")
    assert "plotly" in page.lower()
    assert not re.search(r"<script[^>]*\bsrc=[\"']https?://", page), (
        "the fairness table loads plotly.js from the network; it must open with the wifi off")

"""fairness.py -- who does this model miss? Written before the module.

The audit asks two questions of every group: is the model as well calibrated here as it is
everywhere (ECE), and does it miss as few real events here as it does everywhere (false
negative rate). A group whose relative FNR gap exceeds 20 percent is `flagged`.

These tests exist mostly to make the audit hard to soften. A group is never dropped for
being small, an unknown race is its own group rather than a deleted row, the flag fires on
the ratio and nothing else, and `failed()` is true the moment one row is flagged. The
guardrail in `test_guardrails.py` covers the other half: no swallowed exceptions.
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
    """0 of 0 missed is not a pass. `n_events` is in the table so nobody reads it as one."""
    scores, outcomes, cohort = _boroughs(bronx=[(False, True)] * 4,
                                         queens=[(False, False)] * 4)
    rows = _borough_rows(fair.audit(scores, outcomes, cohort))
    assert rows["Queens"]["n_events"] == 0
    assert rows["Queens"]["fnr"] == 0.0
    assert rows["Queens"]["flagged"] is False


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
    for row in tbl.to_dicts():
        FairnessRow(stratum=row["stratum"], group=row["group"], n=row["n"], ece=row["ece"],
                    fnr=row["fnr"], fnr_ratio_to_cohort=row["fnr_ratio_to_cohort"],
                    flagged=row["flagged"])


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

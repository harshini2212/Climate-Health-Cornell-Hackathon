"""report.py -- the eval harness assembled into `report/report.json`. Written before the module.

`GET /report` serves this file as-is and the Model report screen draws it. So the tests that
matter are about honesty rather than arithmetic: the rung is the rung the scores were made
at, a failing fairness row survives the trip into the JSON, and nothing about the run is
claimed that did not happen -- no r-hat at a rung with no MCMC.
"""

from __future__ import annotations

import json

import polars as pl
import pytest

from leeward.api.schemas import ReportResponse
from leeward.eval import calibration as cal
from leeward.eval import fairness as fair
from leeward.eval import report as rep
from leeward.schema import NEEDS
from tables import table


@pytest.fixture(scope="module")
def report() -> rep.Report:
    """One assembled report over a two-day window -- the whole harness, run once."""
    scores, outcomes, cohort = table("scores"), table("outcomes"), table("cohort")
    days = cal.holdout_dates(scores, outcomes, n_days=2)
    return rep.assemble(scores=scores, outcomes=outcomes, cohort=cohort, dates=days,
                        ks=(20,), n_draws=200)


@pytest.fixture(scope="module")
def payload(report: rep.Report) -> dict:
    return report.payload


def test_the_report_validates_as_the_api_contract(payload: dict) -> None:
    ReportResponse.model_validate(payload)


def test_nothing_is_claimed_that_the_contract_has_no_room_for(payload: dict) -> None:
    """`ReportResponse` forbids extra keys, so a field invented here would 500 `GET /report`
    rather than show up on the screen. Catch it at the source instead."""
    assert set(payload) <= set(ReportResponse.model_fields)


def test_the_rung_is_the_rung_the_scores_were_made_at(payload: dict) -> None:
    assert payload["model_rung"] == table("scores")["model_rung"][0]


def test_scores_that_mix_rungs_are_an_error_not_an_average() -> None:
    mixed = table("scores").with_columns(
        pl.when(pl.int_range(pl.len()) < 10).then(pl.lit(1, dtype=pl.Int32))
          .otherwise(pl.col("model_rung")).alias("model_rung"))
    with pytest.raises(ValueError, match="rung"):
        rep.model_rung(mixed)


def test_a_rung_with_no_mcmc_claims_no_rhat_and_no_divergences(payload: dict) -> None:
    assert payload["model_rung"] == 0
    assert payload["rhat_max"] is None
    assert payload["divergences"] is None


def test_calibration_and_its_ece_are_both_present(payload: dict) -> None:
    assert payload["calibration"], "the calibration plot is on the never-cut list"
    assert set(payload["ece_by_need"]) == set(NEEDS)
    assert {row["need"] for row in payload["calibration"]} == set(NEEDS)


def test_recovery_coverage_is_the_fraction_of_covered_rows(payload: dict) -> None:
    covered = sum(row["covered"] for row in payload["recovery"])
    assert payload["recovery_coverage"] == pytest.approx(covered / len(payload["recovery"]))
    assert payload["recovery_coverage"] >= 0.90, "SPEC §11 pass bar"


def test_decision_quality_carries_every_strategy_as_measured(report: rep.Report) -> None:
    """Every strategy appears with the number `decision_quality` computed, including the
    days a baseline wins. The assembler reports the comparison; it does not curate it --
    see docs/PROMPTS.md 16 for why that matters on the fixture cohort right now."""
    rows = report.payload["decision_quality"]
    assert {row["strategy"] for row in rows} == set(rep.dq.STRATEGIES)
    want = rep.dq.summarise(report.decision_quality)
    assert [r["harm_averted"] for r in rows] == want["harm_averted"].to_list()
    assert [r["strategy"] for r in rows] == want["strategy"].to_list()


def test_the_fairness_table_is_the_audit_row_for_row(report: rep.Report) -> None:
    assert len(report.payload["fairness"]) == report.fairness.height
    assert report.payload["fairness_failed"] is fair.failed(report.fairness)


def test_a_failing_audit_reaches_the_json_rather_than_being_filtered(monkeypatch) -> None:
    """The one thing we will not ship is a report that looks clean because the audit was
    dropped on the way out."""
    def fake_audit(*a, **kw):
        return pl.DataFrame(
            [{"stratum": "borough", "group": "Bronx", "n": 10, "n_events": 5, "n_missed": 5,
              "ece": 0.4, "fnr": 1.0, "fnr_ratio_to_cohort": 2.0, "flagged": True}],
            schema=fair.AUDIT_SCHEMA)

    monkeypatch.setattr(rep.fair, "audit", fake_audit)
    scores, outcomes, cohort = table("scores"), table("outcomes"), table("cohort")
    days = cal.holdout_dates(scores, outcomes, n_days=1)
    out = rep.assemble(scores=scores, outcomes=outcomes, cohort=cohort, dates=days,
                       ks=(20,), n_draws=64).payload
    assert out["fairness_failed"] is True
    assert out["fairness"] == [{"stratum": "borough", "group": "Bronx", "n": 10, "ece": 0.4,
                                "fnr": 1.0, "fnr_ratio_to_cohort": 2.0, "flagged": True}]
    ReportResponse.model_validate(out)


def test_generated_at_is_an_iso_timestamp(payload: dict) -> None:
    from datetime import datetime
    assert datetime.fromisoformat(payload["generated_at"]).year >= 2025


def test_writing_leaves_valid_json_and_the_offline_side_tables(report: rep.Report,
                                                               tmp_path) -> None:
    """`make report` leaves the JSON the API serves and the CSVs and charts the deck uses."""
    paths = report.write(tmp_path)
    back = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    ReportResponse.model_validate(back)
    assert back["fairness_failed"] == report.payload["fairness_failed"]
    assert back["generated_at"] == report.payload["generated_at"]
    for name in ("report.json", "calibration.csv", "calibration.html", "recovery.csv",
                 "recovery.html", "fairness.csv", "fairness.html",
                 "decision_quality.csv", "decision_quality.html"):
        assert (tmp_path / name) in paths and (tmp_path / name).exists(), name

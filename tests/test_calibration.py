"""calibration.py -- reliability per need on the held-out window. Written before the module.

A reliability curve is only worth showing if the bins mean something. Rung 0 predicts a
daily probability in the low percent, so equal-width bins on [0, 1] would drop nearly every
row into the first bin and report an ECE near zero no matter how wrong the model was. These
tests pin the equal-mass binning that avoids that, the ECE arithmetic by hand, and the rule
that a scored veteran-day with no outcome is an error rather than a quiet zero.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from leeward.api.schemas import CalibrationBin
from leeward.eval import calibration as cal
from leeward.schema import NEEDS
from tables import table

D0 = date(2026, 7, 1)


def _pair(rows: list[tuple[str, date, str, float, int]]) -> pl.DataFrame:
    """A hand-built `paired` frame: veteran_id, date, need, p_mean, y."""
    return pl.DataFrame(rows, orient="row",
                        schema={"veteran_id": pl.Utf8, "date": pl.Date, "need": pl.Utf8,
                                "p_mean": pl.Float64, "y": pl.Int32})


def _flat(p: list[float], y: list[int], need: str = "heat") -> pl.DataFrame:
    return _pair([(f"v{i}", D0, need, pi, yi) for i, (pi, yi) in enumerate(zip(p, y, strict=True))])


# --------------------------------------------------------------------------- #
# The held-out window
# --------------------------------------------------------------------------- #

def test_the_holdout_window_is_the_last_n_days_both_tables_share() -> None:
    days = [D0 + timedelta(days=i) for i in range(10)]
    scores = _pair([("v", d, "heat", 0.1, 0) for d in days])
    outcomes = _pair([("v", d, "heat", 0.1, 0) for d in days[:8]])
    assert cal.holdout_dates(scores, outcomes, n_days=3) == days[5:8]


def test_a_window_with_no_shared_days_is_an_error() -> None:
    scores = _pair([("v", D0, "heat", 0.1, 0)])
    outcomes = _pair([("v", D0 + timedelta(days=99), "heat", 0.1, 0)])
    with pytest.raises(ValueError, match="share no dates"):
        cal.holdout_dates(scores, outcomes)


# --------------------------------------------------------------------------- #
# Pairing scores to what happened
# --------------------------------------------------------------------------- #

def test_pairing_keeps_one_row_per_scored_veteran_day_need() -> None:
    scores = table("scores")
    outcomes = table("outcomes")
    days = cal.holdout_dates(scores, outcomes, n_days=2)
    paired = cal.paired(scores, outcomes, dates=days)
    assert paired.columns == ["veteran_id", "date", "need", "p_mean", "y"]
    assert paired.height == scores.filter(pl.col("date").is_in(days)).height
    assert set(paired["date"].unique().to_list()) == set(days)
    assert paired["y"].null_count() == 0


def test_a_scored_veteran_day_with_no_outcome_is_an_error_not_a_zero() -> None:
    """Scoring a missing outcome as 0 would flatter the model exactly where it is blind."""
    scores = _pair([("v1", D0, "heat", 0.3, 0), ("ghost", D0, "heat", 0.9, 0)])
    outcomes = _pair([("v1", D0, "heat", 0.0, 1)]).select("veteran_id", "date", "need", "y")
    with pytest.raises(ValueError, match="ghost"):
        cal.paired(scores, outcomes)


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #

def test_ece_is_hand_checkable_on_a_single_bin() -> None:
    """Twenty rows all predicted 0.2; ten of them happened. ECE = |0.5 - 0.2| = 0.3."""
    rel = cal.reliability(_flat([0.2] * 20, [1] * 10 + [0] * 10))
    assert rel.height == 1, "identical predictions cannot be split into several bins"
    row = rel.row(0, named=True)
    assert (row["predicted"], row["observed"], row["n"]) == (pytest.approx(0.2),
                                                             pytest.approx(0.5), 20)
    assert cal.ece(rel)["heat"] == pytest.approx(0.3)


def test_a_perfectly_calibrated_predictor_has_zero_ece() -> None:
    p = [0.0] * 50 + [0.5] * 50 + [1.0] * 50
    y = [0] * 50 + [1] * 25 + [0] * 25 + [1] * 50
    assert cal.ece(cal.reliability(_flat(p, y)))["heat"] == pytest.approx(0.0, abs=1e-12)


def test_ece_is_weighted_by_how_many_rows_fall_in_each_bin() -> None:
    """A big well-calibrated bin must not be outvoted by a small badly-calibrated one."""
    p = [0.1] * 90 + [0.9] * 10
    y = [1] * 9 + [0] * 81 + [0] * 10          # bin 1: 0.1 observed. bin 2: 0.0 vs 0.9.
    rel = cal.reliability(_flat(p, y), n_bins=2)
    assert rel.height == 2
    assert cal.ece(rel)["heat"] == pytest.approx(0.9 * 0.0 + 0.1 * 0.9)


def test_ece_is_reported_per_need_not_pooled() -> None:
    rows = [(f"v{i}", D0, "heat", 0.2, 1) for i in range(10)]
    rows += [(f"v{i}", D0, "mental", 0.2, 0) for i in range(10)]
    by_need = cal.ece(cal.reliability(_pair(rows)))
    assert by_need == {"heat": pytest.approx(0.8), "mental": pytest.approx(0.2)}


def test_overall_ece_pools_every_need_by_row_count() -> None:
    rows = [(f"v{i}", D0, "heat", 0.2, 1) for i in range(30)]
    rows += [(f"v{i}", D0, "mental", 0.2, 0) for i in range(10)]
    rel = cal.reliability(_pair(rows))
    assert cal.ece_overall(rel) == pytest.approx((30 * 0.8 + 10 * 0.2) / 40)


# --------------------------------------------------------------------------- #
# The binning, which is the part that can silently lie
# --------------------------------------------------------------------------- #

def test_bins_hold_equal_numbers_of_rows_not_equal_widths() -> None:
    """Rung 0's daily risks are small and skewed. Equal-width bins would report one bin
    holding 99% of the rows and an ECE that says nothing about the other 1%."""
    p = list(np.logspace(-4, np.log10(0.5), 1000))
    rel = cal.reliability(_flat(p, [0] * 1000), n_bins=10)
    assert rel.height == 10
    assert rel["n"].min() >= 90 and rel["n"].max() <= 110
    widths = np.diff(rel["predicted"].to_numpy())
    assert widths.max() > 5 * widths.min(), "bins should be uneven in width, even in mass"


def test_bins_are_ordered_and_predicted_rises_with_them() -> None:
    p = list(np.linspace(0.01, 0.6, 500))
    rel = cal.reliability(_flat(p, [0] * 500), n_bins=5)
    assert rel["bin"].to_list() == sorted(rel["bin"].to_list())
    assert rel["predicted"].to_list() == sorted(rel["predicted"].to_list())


def test_shared_edges_make_two_groups_comparable() -> None:
    """The fairness audit bins every group on the whole cohort's edges, so that a gap in ECE
    is a gap in the model and not a gap in where the bin boundaries happened to fall."""
    everyone = _flat(list(np.linspace(0.0, 1.0, 100)), [0] * 100)
    edges = cal.bin_edges(everyone, n_bins=4)
    # Strictly above the middle edge (0.5), so the subset cannot reach the lower two bins.
    top_half = _flat(list(np.linspace(0.51, 1.0, 50)), [0] * 50)
    rel = cal.reliability(top_half, edges=edges)
    assert set(rel["bin"].to_list()) == {2, 3}, "a subset must land in the cohort's own bins"


def test_every_need_keeps_its_own_edges() -> None:
    """Needs sit at different base rates; one shared set of edges would empty most bins."""
    rows = [(f"v{i}", D0, "heat", 0.001 * i, 0) for i in range(100)]
    rows += [(f"v{i}", D0, "mental", 0.5 + 0.001 * i, 0) for i in range(100)]
    rel = cal.reliability(_pair(rows), n_bins=4)
    assert rel.filter(pl.col("need") == "heat").height == 4
    assert rel.filter(pl.col("need") == "mental").height == 4


# --------------------------------------------------------------------------- #
# What comes out
# --------------------------------------------------------------------------- #

def test_reliability_rows_fit_the_report_contract() -> None:
    rel = cal.reliability(cal.paired(table("scores"), table("outcomes")))
    assert set(rel["need"].unique().to_list()) == set(NEEDS)
    for row in rel.to_dicts():
        CalibrationBin(need=row["need"], predicted=row["predicted"],
                       observed=row["observed"], n=row["n"])


def test_outputs_are_a_tidy_csv_and_an_offline_plotly_chart(tmp_path) -> None:
    rel = cal.reliability(cal.paired(table("scores"), table("outcomes")))
    csv, html = cal.write_outputs(rel, tmp_path)
    back = pl.read_csv(csv)
    assert back.columns == list(cal.RELIABILITY_SCHEMA)
    assert back.height == rel.height
    page = html.read_text(encoding="utf-8")
    assert "plotly" in page.lower()
    assert not re.search(r"<script[^>]*\bsrc=[\"']https?://", page), (
        "the calibration plot loads plotly.js from the network; it must open with the wifi off")


# --------------------------------------------------------------------------- #
# The control: what a predictor that has never met anyone scores here
# --------------------------------------------------------------------------- #

def test_a_constant_at_the_base_rate_scores_zero_ece_exactly() -> None:
    """The finding, asserted rather than hoped for. P(Y | f(X) = π) = π by definition, so the
    calibration error of a constant is zero for any data -- not a quirk of these bins."""
    pairs = _flat([0.02, 0.9, 0.5, 0.11] * 25, ([1] + [0] * 3) * 25)
    assert cal.constant_ece(pairs)["heat"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n_bins", [2, 3, 5, 10, 50])
def test_the_control_scores_zero_at_every_bin_count(n_bins: int) -> None:
    """The load-bearing version of the claim. If this were a binning artifact, some bin count
    would break it and the honest fix would be "use more bins" or "use a smoother". No bin
    count breaks it, because calibration error is proper but not *strictly* proper: it drops
    the sharpness term, so a constant at the base rate minimises it as surely as the truth
    does. This is why the answer is discrimination and not a better calibration metric."""
    pairs = _flat(list(np.linspace(0.001, 0.6, 400)), [1] * 37 + [0] * 363)
    assert cal.constant_ece(pairs, n_bins=n_bins)["heat"] == pytest.approx(0.0, abs=1e-12)


def test_the_control_is_one_bin_holding_the_base_rate() -> None:
    rel = cal.reliability(cal.constant_pairs(_flat([0.02, 0.9] * 50, [1] * 20 + [0] * 80)))
    assert rel.height == 1, "a constant predictor cannot be split into several bins"
    row = rel.row(0, named=True)
    assert row["predicted"] == pytest.approx(0.2)
    assert row["observed"] == pytest.approx(0.2)
    assert row["n"] == 100


def test_the_control_uses_each_needs_own_base_rate() -> None:
    """One number per need, not one across the cohort: heat happens seven times as often as
    access_loss, and a shared constant would be miscalibrated on both."""
    rows = [(f"v{i}", D0, "heat", 0.3, int(i < 40)) for i in range(100)]
    rows += [(f"v{i}", D0, "mental", 0.3, int(i < 5)) for i in range(100)]
    got = cal.constant_pairs(_pair(rows))
    assert got.filter(pl.col("need") == "heat")["p_mean"].unique().to_list() == [0.4]
    assert got.filter(pl.col("need") == "mental")["p_mean"].unique().to_list() == [0.05]
    assert cal.constant_ece(_pair(rows)) == {"heat": pytest.approx(0.0),
                                            "mental": pytest.approx(0.0)}


def test_the_control_is_at_or_below_the_models_ece_on_every_need() -> None:
    """This is the uncomfortable half of the calibration claim and the reason the control
    ships. If a change ever makes the model beat a constant on ECE, this test goes red and
    somebody gets to delete the caveat -- deliberately."""
    pairs = cal.paired(table("scores"), table("outcomes"))
    model, control = cal.ece(cal.reliability(pairs)), cal.constant_ece(pairs)
    assert set(control) == set(NEEDS)
    for need in NEEDS:
        assert control[need] <= model[need] + 1e-12, (
            f"{need}: a constant at the base rate scores {control[need]:.4f} against the "
            f"model's {model[need]:.4f}; the caveat in the docstring may be stale")


def test_base_rates_are_recoverable_from_the_bins_alone() -> None:
    """The chart places the control without being handed the pairs again, so the bins have
    to carry the base rate: Σ(n · observed) / Σ n."""
    pairs = cal.paired(table("scores"), table("outcomes"))
    recovered = cal.base_rates(cal.reliability(pairs))
    for need, rate in recovered.items():
        direct = pairs.filter(pl.col("need") == need)["y"].mean()
        assert rate == pytest.approx(direct)


def test_the_reliability_chart_labels_the_control_on_the_diagonal() -> None:
    """A reviewer's first question is "what would a constant score?". The answer is on the
    chart, named, rather than in a pause."""
    rel = cal.reliability(cal.paired(table("scores"), table("outcomes")))
    fig = cal.reliability_curve(rel)
    control = [tr for tr in fig.data if tr.name and "constant" in tr.name.lower()]
    assert len(control) == 1, "the constant-predictor control must appear exactly once"
    assert "0.000" in control[0].name
    # On the diagonal by construction: predicted == observed == the base rate.
    assert list(control[0].x) == pytest.approx(list(control[0].y))
    assert sorted(control[0].x) == pytest.approx(sorted(cal.base_rates(rel).values()))


def test_the_fixture_cohort_is_close_to_calibrated() -> None:
    """The fixture scores are drawn to sit near their own event rate. This is a smoke test on
    the pipeline, not a claim about the model: a real ECE comes from `make report`."""
    rel = cal.reliability(cal.paired(table("scores"), table("outcomes")))
    assert cal.ece_overall(rel) < 0.15

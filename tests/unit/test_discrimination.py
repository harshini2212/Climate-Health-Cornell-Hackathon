"""discrimination.py -- can the model tell today's sick veteran from today's well one?

Written before the module, and written because the calibration screen cannot answer the
question. At base rates of a third of a percent to two percent, a single number equal to the
base rate scores a perfect ECE (see `test_calibration.py`), so calibration alone cannot tell
Leeward apart from a predictor that has never met anyone. Discrimination can.

The load-bearing test here is `test_within_day_auc_is_below_pooled_when_the_day_is_the_signal`.
The call list is chosen *within* a day, so pooled AUC is partly credit for knowing that today
is a heat wave -- which is not a decision anyone makes. These tests pin that the two numbers
come apart, on a frame built so that both are hand-checkable, rather than on whatever
`data/scores.parquet` happens to hold (see `tables.py` for why the suite never reads it).
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from leeward.api.schemas import DiscriminationRow
from leeward.eval import calibration as cal
from leeward.eval import discrimination as disc
from leeward.schema import NEEDS
from tables import table

D0 = date(2026, 7, 1)


def _pair(rows: list[tuple[str, date, str, float, int]]) -> pl.DataFrame:
    """A hand-built `paired` frame: veteran_id, date, need, p_mean, y."""
    return pl.DataFrame(rows, orient="row",
                        schema={"veteran_id": pl.Utf8, "date": pl.Date, "need": pl.Utf8,
                                "p_mean": pl.Float64, "y": pl.Int32})


def _day(p: list[float], y: list[int], day: date = D0, need: str = "heat") -> pl.DataFrame:
    return _pair([(f"v{i}", day, need, pi, yi)
                  for i, (pi, yi) in enumerate(zip(p, y, strict=True))])


# --------------------------------------------------------------------------- #
# AUC, by the rank identity
# --------------------------------------------------------------------------- #

def test_a_perfectly_ranked_predictor_scores_auc_one() -> None:
    p = list(np.linspace(0.01, 0.99, 200))
    y = [0] * 100 + [1] * 100                  # every event ranked above every non-event
    assert disc.auc(p, y) == pytest.approx(1.0)


def test_a_shuffled_predictor_scores_auc_one_half() -> None:
    """Seeded, so this is the same number in rehearsal and on stage."""
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 2000)
    p = rng.permutation(np.linspace(0.01, 0.99, y.size))
    assert disc.auc(p, y) == pytest.approx(0.5, abs=0.02)


def test_a_perfectly_wrong_predictor_scores_auc_zero() -> None:
    assert disc.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)


def test_ties_are_averaged_not_broken_by_row_order() -> None:
    """A tie is half a win. Breaking it by whichever row came first would let the cohort's
    arrival order move the headline number."""
    assert disc.auc([0.5] * 100, [1] * 50 + [0] * 50) == pytest.approx(0.5)
    # One positive at 0.7; negatives at 0.7 and 0.1. Tie -> 0.5, win -> 1. U = 1.5 of 2.
    assert disc.auc([0.7, 0.7, 0.1], [1, 0, 0]) == pytest.approx(0.75)
    assert disc.auc([0.1, 0.7, 0.7], [0, 0, 1]) == pytest.approx(0.75)


def test_auc_is_undefined_with_no_events_rather_than_zero() -> None:
    """Nothing happened all window: there is no ranking question to answer. Returning 0.0
    would read as "the model got every pair backwards"."""
    assert disc.auc([0.3, 0.2, 0.1], [0, 0, 0]) is None
    assert disc.auc([0.3, 0.2, 0.1], [1, 1, 1]) is None


# --------------------------------------------------------------------------- #
# Within-day against pooled -- the reason this module exists
# --------------------------------------------------------------------------- #

def test_within_day_auc_is_below_pooled_when_the_day_is_the_signal() -> None:
    """Two days. Within either one the score is flat, so it ranks nobody: within-day AUC is
    exactly 0.5. Pooled, the score separates the hot day from the mild one and the hot day
    carries more events, so pooled AUC is 0.78 -- all of it credit for knowing the weather.

    Hand-check of the pooled number, tie-averaged over 55 x 145 pairs:
        50x50 tied at 0.5 -> 1250 · 50x95 won -> 4750 · 5x50 lost -> 0 · 5x95 tied -> 237.5
        (1250 + 4750 + 237.5) / 7975 = 0.78213
    """
    hot = _day([0.5] * 100, [1] * 50 + [0] * 50, day=D0)
    mild = _day([0.05] * 100, [1] * 5 + [0] * 95, day=D0 + timedelta(days=1))
    pairs = pl.concat([hot, mild])

    assert disc.within_day_auc(pairs) == pytest.approx(0.5)
    assert disc.auc(pairs["p_mean"], pairs["y"]) == pytest.approx(6237.5 / 7975)
    row = disc.discriminate(pairs).row(0, named=True)
    assert row["within_day_auc"] < row["pooled_auc"]
    assert row["n_days"] == 2


def test_within_day_auc_is_the_mean_over_days_not_pooled_over_rows() -> None:
    """A quiet day counts as much as a busy one: the care team has to choose a list on both."""
    good = _day(list(np.linspace(0.01, 0.99, 100)), [0] * 50 + [1] * 50, day=D0)
    coin = _day([0.5] * 400, [1] * 200 + [0] * 200, day=D0 + timedelta(days=1))
    # Pooling rows would let the 400-row day drown the 100-row one; the mean of days is 0.75.
    assert disc.within_day_auc(pl.concat([good, coin])) == pytest.approx(0.75)


def test_days_with_nothing_to_rank_are_skipped_not_scored_as_a_half() -> None:
    """A day where nobody had an event has no AUC. Folding it in as 0.5 would drag every
    rare need toward a coin flip and call that the model's fault.

    But dropping it conditions on the outcome, so the count of days dropped is reported
    rather than left off: `n_days` of `n_days_total`."""
    real = _day(list(np.linspace(0.01, 0.99, 100)), [0] * 50 + [1] * 50, day=D0)
    empty = _day([0.4] * 100, [0] * 100, day=D0 + timedelta(days=1))
    pairs = pl.concat([real, empty])
    assert disc.within_day_auc(pairs) == pytest.approx(1.0)
    row = disc.discriminate(pairs).row(0, named=True)
    assert (row["n_days"], row["n_days_total"]) == (1, 2), (
        "a mean over an unstated subset of days is the kind of number that wins a demo and "
        "loses a reviewer; both counts ship")


def test_within_day_auc_is_undefined_when_no_single_day_can_be_scored() -> None:
    assert disc.within_day_auc(_day([0.4] * 50, [0] * 50)) is None


# --------------------------------------------------------------------------- #
# PR-AUC, which is the number that respects the base rate
# --------------------------------------------------------------------------- #

def test_pr_auc_of_a_perfect_ranker_is_one() -> None:
    p = list(np.linspace(0.01, 0.99, 200))
    assert disc.average_precision(p, [0] * 100 + [1] * 100) == pytest.approx(1.0)


def test_pr_auc_of_an_uninformative_score_is_about_the_base_rate() -> None:
    """The floor PR-AUC is measured against, and the reason it is reported next to AUC:
    at a 1 percent base rate a PR-AUC of 0.05 is five times chance, and an AUC of 0.6 does
    not tell you that."""
    rng = np.random.default_rng(1)
    y = (rng.random(20_000) < 0.01).astype(int)
    p = rng.random(20_000)
    assert disc.average_precision(p, y) == pytest.approx(0.01, abs=0.005)


def test_pr_auc_holds_tied_scores_together() -> None:
    """Every score identical: precision is the base rate at every recall, so PR-AUC is the
    base rate exactly. A tie-break would invent a ranking and report up to 1.0."""
    assert disc.average_precision([0.3] * 100, [1] * 20 + [0] * 80) == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# Scaled Brier -- the score on which the constant does NOT win
# --------------------------------------------------------------------------- #

def test_the_constant_that_wins_on_ece_scores_exactly_zero_here() -> None:
    """The whole point of carrying a strictly proper score. The same predictor that scores a
    perfect ECE (see test_calibration.py) scores 0.0 skill: no better than knowing nothing."""
    y = [1] * 23 + [0] * 977
    base = 0.023
    assert disc.scaled_brier([base] * 1000, y) == pytest.approx(0.0, abs=1e-12)


def test_a_perfect_predictor_scores_one_and_a_bad_one_goes_negative() -> None:
    y = [1] * 30 + [0] * 70
    assert disc.scaled_brier([1.0] * 30 + [0.0] * 70, y) == pytest.approx(1.0)
    # Confidently backwards: worse than knowing nothing, and the score says so rather than
    # bottoming out at zero.
    assert disc.scaled_brier([0.0] * 30 + [1.0] * 70, y) < 0


def test_the_null_brier_is_the_closed_form_base_rate_variance() -> None:
    """Brier_null = π(1-π). Pinned because scaled_brier divides by it, and a wrong reference
    would silently rescale every skill number on the screen."""
    y = [1] * 200 + [0] * 800
    assert disc.brier([0.2] * 1000, y) == pytest.approx(0.2 * 0.8)


def test_the_raw_brier_flatters_a_rare_event_and_the_scaled_one_does_not() -> None:
    """Predicting zero for everybody scores a Brier of 0.0034 at a 0.34% base rate, which
    looks excellent and means nothing. That is why both are reported."""
    y = [1] * 34 + [0] * 9966
    assert disc.brier([0.0] * 10_000, y) == pytest.approx(0.0034)
    assert disc.scaled_brier([0.0] * 10_000, y) < 0


def test_skill_is_undefined_when_there_is_none_to_have() -> None:
    assert disc.scaled_brier([0.1] * 50, [0] * 50) is None
    assert disc.scaled_brier([0.9] * 50, [1] * 50) is None


def test_the_fixture_scores_show_positive_skill() -> None:
    """A smoke test on the arithmetic, **not** a claim about Leeward. The fixture draws
    outcomes as Bernoulli(p_mean), so its scores are perfectly calibrated *and* sharp and
    skill is positive by construction; if this went negative the estimator would be broken.

    The real rung-0 model is not like this and the suite must not imply otherwise: on
    `make report` heat scores +0.110 and treatment_gap +0.042, while access_loss (-0.014),
    mental (-0.004) and breathing (+0.001) sit at or below the constant. Those three rank
    above chance (AUC 0.58-0.65) but their probability scale does not yet beat the base rate
    on a proper score -- which is what prior-only means, and where a rung-1 fit would pay.
    """
    got = disc.discriminate(cal.paired(table("scores"), table("outcomes")))
    assert got["scaled_brier"].min() > 0.0


# --------------------------------------------------------------------------- #
# Lift -- what the top of the call list actually buys
# --------------------------------------------------------------------------- #

def test_lift_at_one_percent_is_hand_checkable() -> None:
    """1,000 veterans, 50 events, and the ten highest-scored all had one. The top 1 percent
    is 10 calls at a 100 percent hit rate against a 5 percent base rate: 20x."""
    p = list(np.linspace(0.99, 0.01, 1000))
    y = [1] * 10 + [0] * 950 + [1] * 40
    assert disc.lift(_day(p, y), 0.01) == pytest.approx(20.0)


def test_lift_of_an_uninformative_score_is_about_one() -> None:
    rng = np.random.default_rng(2)
    p = list(rng.random(5000))
    y = list((rng.random(5000) < 0.1).astype(int))
    assert disc.lift(_day(p, y), 0.10) == pytest.approx(1.0, abs=0.15)


def test_lift_selects_within_each_day_because_that_is_when_the_list_is_made() -> None:
    """The team gets a budget every morning, not one budget for the month. On the mild day
    the top 10 percent are the day's own worst 10, not nobody."""
    hot = _day(list(np.linspace(0.90, 0.50, 100)), [1] * 20 + [0] * 80, day=D0)
    mild = _day(list(np.linspace(0.09, 0.01, 100)), [1] * 10 + [0] * 90,
                day=D0 + timedelta(days=1))
    pairs = pl.concat([hot, mild])
    # Per day: 10 from each day, all 20 of them events. Base rate 30/200 = 0.15 -> 6.67x.
    # Pooling the window first would take 20 rows from the hot day only and score 20/20 too,
    # but on the mild day it would have called nobody.
    assert disc.lift(pairs, 0.10) == pytest.approx(1.0 / 0.15)


def test_lift_ties_break_on_veteran_id_so_the_same_click_gives_the_same_number() -> None:
    p = [0.2] * 100
    y = [1] * 10 + [0] * 90
    pairs = _day(p, y)
    assert disc.lift(pairs, 0.10) == disc.lift(pairs.sample(fraction=1.0, shuffle=True, seed=7),
                                               0.10)


def test_lift_is_undefined_with_no_events() -> None:
    assert disc.lift(_day([0.3] * 100, [0] * 100), 0.01) is None


def test_lift_never_exceeds_its_ceiling() -> None:
    """A perfect ranker at a 2% base rate still cannot beat 50x on the top 1%: there are not
    enough events to fill the list. Reporting lift without the ceiling hides that."""
    pairs = _day(list(np.linspace(0.99, 0.01, 1000)), [1] * 20 + [0] * 980)
    assert disc.lift_ceiling(pairs, 0.01) == pytest.approx(min(100.0, 1 / 0.02))
    assert disc.lift(pairs, 0.01) <= disc.lift_ceiling(pairs, 0.01) + 1e-9


def test_the_ceiling_is_the_tighter_of_the_two_bounds() -> None:
    """min(1/q, 1/base): at a 0.34% base rate the top 1% is capped by the slice (100x), and
    at a 2.33% base rate it is capped by how few events there are (42.9x)."""
    rare = _day([0.5] * 1000, [1] * 3 + [0] * 997)            # 0.3% base
    common = _day([0.5] * 1000, [1] * 233 + [0] * 767)        # 23.3% base
    assert disc.lift_ceiling(rare, 0.01) == pytest.approx(100.0)
    assert disc.lift_ceiling(common, 0.01) == pytest.approx(1 / 0.233)


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

def test_discriminate_reports_one_row_per_need_in_contract_order() -> None:
    pairs = cal.paired(table("scores"), table("outcomes"))
    got = disc.discriminate(pairs)
    assert got["need"].to_list() == list(NEEDS)
    assert got.columns == list(disc.DISCRIMINATION_SCHEMA)


def test_discriminate_counts_what_it_scored() -> None:
    pairs = cal.paired(table("scores"), table("outcomes"))
    got = disc.discriminate(pairs)
    for row in got.to_dicts():
        part = pairs.filter(pl.col("need") == row["need"])
        assert row["n"] == part.height
        assert row["n_events"] == int(part["y"].sum())
        assert row["base_rate"] == pytest.approx(row["n_events"] / row["n"])


def test_an_informative_fixture_beats_a_coin_flip_on_every_need() -> None:
    """The fixture draws outcomes as Bernoulli(p_mean), so its scores really do rank people.
    A module that reported 0.5 here would be broken, not modest. This is a smoke test on the
    arithmetic; the model's own numbers come from `make report`."""
    got = disc.discriminate(cal.paired(table("scores"), table("outcomes")))
    assert got["within_day_auc"].min() > 0.6
    assert got["pooled_auc"].min() > 0.6
    assert got["lift_at_1pct"].min() > 1.0


def test_discrimination_rows_fit_the_report_contract() -> None:
    for row in disc.discriminate(cal.paired(table("scores"), table("outcomes"))).to_dicts():
        DiscriminationRow(**row)


def test_run_scores_only_the_held_out_window() -> None:
    scores, outcomes = table("scores"), table("outcomes")
    days = cal.holdout_dates(scores, outcomes, n_days=3)
    got = disc.run(scores, outcomes, dates=days)
    assert got["n_days"].max() == 3
    assert got["n"].sum() == scores.filter(pl.col("date").is_in(days)).height


def test_outputs_are_a_tidy_csv_and_an_offline_plotly_chart(tmp_path) -> None:
    got = disc.discriminate(cal.paired(table("scores"), table("outcomes")))
    csv, html = disc.write_outputs(got, tmp_path)
    back = pl.read_csv(csv)
    assert back.columns == list(disc.DISCRIMINATION_SCHEMA)
    assert back.height == got.height
    page = html.read_text(encoding="utf-8")
    assert "plotly" in page.lower()
    assert not re.search(r"<script[^>]*\bsrc=[\"']https?://", page), (
        "the discrimination chart loads plotly.js from the network; it must open offline")


def test_the_chart_shows_within_day_and_pooled_side_by_side() -> None:
    """Pooled alone is the flattering number. If it ever ships without within-day beside it,
    this test is the thing that notices."""
    fig = disc.auc_chart(disc.discriminate(cal.paired(table("scores"), table("outcomes"))))
    names = [tr.name for tr in fig.data if tr.name]
    assert any("within-day" in n.lower() for n in names)
    assert any("pooled" in n.lower() for n in names)

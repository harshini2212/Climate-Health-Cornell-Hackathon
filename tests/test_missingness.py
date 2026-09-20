"""Acceptance tests for `leeward/cohort/missingness.py` and the marginalisation it forces
on `leeward/model/score_prior.py`.

Written before either module. Two claims are protected here, and they are different claims:

1. **The record has holes, and only scoring is blinded by them.** The truth stays in the
   cohort because `cohort/simulate.py` draws outcomes from it -- the world knows whether a
   veteran has air conditioning even when the VA does not. `test_scoring_cannot_read_the_truth`
   is the one that would catch a scorer quietly reading the wrong column, and it is the whole
   reason the `_observed` split is worth having.

2. **A gap widens the interval for the person who has it.** Hidden field -> the value is drawn
   from its population rate once per posterior draw -> that variance lands in `Var(p)`, which
   is the numerator of `p_epistemic_share`. `test_hiding_a_field_widens_that_veterans_interval`
   is the paired form of this and is the sharp one: the same veterans, scored twice, differing
   only in whether their own field was on file.

3. **A gap the care team would act on differently earns a call.** That is `p_gap_lo` and
   `p_gap_hi` -- the number this veteran would be reported with if the missing field turned
   out well or badly -- and the find-out tier firing when that range straddles a line.

The second claim is true and, on its own, nowhere near enough: widening the interval moved
`find_out_share` from 0.226% to 0.231%, because at rung 0 `p_epistemic_share` is dominated by
the prior spread every veteran shares and tops out at 0.426 against a 0.4 cut. The third claim
is what makes the tier demoable, and it is a different measurement of the same gap rather than
a looser threshold on the old one -- which is why
`test_the_gap_rule_does_not_take_veterans_from_act_now` exists. Lowering the old cut to reach
the same volume would have emptied act-now; see the note in `leeward/decision/tiers.py`.
"""

from __future__ import annotations

import polars as pl
import pytest

import tables
from leeward import schema
from leeward.cohort import missingness
from leeward.decision import tiers
from leeward.model import score_prior

DRAWS = 200          # enough for a stable variance, quick enough for the gate
SEED = 0


@pytest.fixture(scope="module")
def cohort() -> pl.DataFrame:
    return tables.table("cohort")


@pytest.fixture(scope="module")
def frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    return tables.table("hazards"), tables.table("site_status")


def _revealed(cohort: pl.DataFrame) -> pl.DataFrame:
    """The same cohort with nothing hidden: every `_observed` column set to the truth."""
    return cohort.with_columns(
        pl.col(f).alias(missingness.observed_name(f)) for f in missingness.HIDDEN_FIELDS)


# --------------------------------------------------------------------------- #
# The gaps themselves
# --------------------------------------------------------------------------- #

def test_every_hidden_field_has_an_observed_sibling(cohort: pl.DataFrame) -> None:
    for field in missingness.HIDDEN_FIELDS:
        assert missingness.observed_name(field) in cohort.columns, f"{field} has no copy"


def test_about_a_fifth_of_each_field_is_hidden(cohort: pl.DataFrame) -> None:
    for field in missingness.HIDDEN_FIELDS:
        share = cohort[missingness.observed_name(field)].null_count() / cohort.height
        assert abs(share - missingness.HIDDEN_FRACTION) < 0.05, (
            f"{field} is {share:.1%} hidden; SPEC 5.5 asks for "
            f"{missingness.HIDDEN_FRACTION:.0%}")


def test_observed_equals_the_truth_wherever_it_is_not_hidden(cohort: pl.DataFrame) -> None:
    """A gap may hide a value. It may never change one."""
    for field in missingness.HIDDEN_FIELDS:
        obs = missingness.observed_name(field)
        disagree = cohort.filter(
            pl.col(obs).is_not_null() & (pl.col(obs) != pl.col(field))).height
        assert disagree == 0, f"{obs} disagrees with {field} on {disagree} rows"


def test_the_truth_column_is_never_nulled(cohort: pl.DataFrame) -> None:
    """`cohort/simulate.py` draws outcomes from these; a null here is a silent outage."""
    for field in missingness.HIDDEN_FIELDS:
        assert cohort[field].null_count() == 0, f"{field} lost its truth"


def test_gaps_are_independent_across_fields(cohort: pl.DataFrame) -> None:
    """Hidden on separate streams, so the same fifth is not missing three times over."""
    hidden = cohort.select(
        pl.col(missingness.observed_name(f)).is_null().alias(f)
        for f in missingness.HIDDEN_FIELDS)
    n_gaps = hidden.sum_horizontal()
    assert n_gaps.max() >= 2, "no veteran is missing two fields; the streams are shared"
    complete = (n_gaps == 0).mean()
    expected = (1 - missingness.HIDDEN_FRACTION) ** len(missingness.HIDDEN_FIELDS)
    assert abs(complete - expected) < 0.07, (
        f"{complete:.1%} of veterans have a complete record, expected about {expected:.1%}")


def test_hiding_is_seeded(cohort: pl.DataFrame) -> None:
    """The same click gives the same gaps in rehearsal and on stage."""
    base = cohort.drop([missingness.observed_name(f) for f in missingness.HIDDEN_FIELDS])
    again = missingness.apply(base, seed=tables.SEED)
    for field in missingness.HIDDEN_FIELDS:
        obs = missingness.observed_name(field)
        assert again[obs].is_null().to_list() == cohort[obs].is_null().to_list()


# --------------------------------------------------------------------------- #
# What the scorer may look at
# --------------------------------------------------------------------------- #

def test_scoring_cannot_read_the_truth(cohort: pl.DataFrame, frames) -> None:
    """Flip the truth under every hidden value and the scores must not move.

    This is the test that matters. Scoring a veteran off a column the VA does not have would
    make the whole exercise a simulation of a system nobody can run, and it would do it
    silently: the numbers would look better, not worse.
    """
    hazards, sites = frames
    days = sorted(hazards["date"].unique().to_list())[:7]
    honest = score_prior.score(cohort, hazards, sites, dates=days, n_draws=DRAWS, seed=SEED)

    # Where the value is hidden, replace the truth with something else entirely.
    lied = cohort.with_columns(
        pl.when(pl.col(missingness.observed_name("home_ac")).is_null())
          .then(~pl.col("home_ac")).otherwise(pl.col("home_ac")).alias("home_ac"),
        pl.when(pl.col(missingness.observed_name("floor")).is_null())
          .then(pl.lit("basement")).otherwise(pl.col("floor")).alias("floor"),
    )
    assert lied["home_ac"].to_list() != cohort["home_ac"].to_list(), "nothing was flipped"
    after = score_prior.score(lied, hazards, sites, dates=days, n_draws=DRAWS, seed=SEED)

    for col in ("p_mean", "p_lo80", "p_hi80", "p_epistemic_share"):
        assert honest[col].to_list() == pytest.approx(after[col].to_list()), (
            f"{col} moved when only the hidden truth changed -- scoring read a column the "
            f"VA does not have")


# --------------------------------------------------------------------------- #
# The claim the find-out tier rests on
# --------------------------------------------------------------------------- #

def test_hiding_a_field_widens_that_veterans_interval(cohort: pl.DataFrame, frames) -> None:
    """The paired form: the same veterans, on the same days, differing only in whether their
    own AC and floor were on file. Every one of them must come out less certain, not more."""
    hazards, sites = frames
    # A week containing a hot day, since `home_ac` reaches the model only through one.
    hot = hazards.filter(pl.col("hot_day"))["date"].unique().sort().to_list()
    assert hot, "the fixture week has no hot day; this test would prove nothing"
    days = hot[:5]

    hidden = score_prior.score(cohort, hazards, sites, dates=days,
                               n_draws=DRAWS, seed=SEED)
    revealed = score_prior.score(_revealed(cohort), hazards, sites, dates=days,
                                 n_draws=DRAWS, seed=SEED)

    gap = cohort.select(
        "veteran_id",
        pl.any_horizontal(pl.col(missingness.observed_name(f)).is_null()
                          for f in ("home_ac", "floor")).alias("has_gap"))
    joined = (hidden.join(revealed, on=["veteran_id", "date", "need"], suffix="_seen")
                    .join(gap, on="veteran_id"))

    with_gap = joined.filter("has_gap")
    assert with_gap.height, "nobody in the fixture cohort has a gap"

    widened = with_gap.select(
        (pl.col("p_epistemic_share") - pl.col("p_epistemic_share_seen")).alias("d"))["d"]
    assert widened.mean() > 0, (
        f"hiding a field did not widen the interval on average (mean delta {widened.mean():.2g})")

    # And a veteran whose record is complete must be scored identically either way.
    unchanged = joined.filter(~pl.col("has_gap")).select(
        (pl.col("p_epistemic_share") - pl.col("p_epistemic_share_seen")).abs().max()).item()
    assert unchanged == pytest.approx(0, abs=1e-12), (
        "a veteran with nothing hidden was scored differently; the marginalisation is "
        "leaking into records that have no gap")


def test_veterans_with_a_gap_are_less_certain_than_veterans_without(
        cohort: pl.DataFrame, frames) -> None:
    """The population form the pitch states: hidden field -> strictly higher mean share.

    Weaker than the paired test above, because who has a gap is not independent of who is
    frail, but it is the number said out loud, so it is checked out loud.
    """
    hazards, sites = frames
    hot = hazards.filter(pl.col("hot_day"))["date"].unique().sort().to_list()[:5]
    scores = score_prior.score(cohort, hazards, sites, dates=hot, n_draws=DRAWS, seed=SEED)

    gap = cohort.select(
        "veteran_id",
        pl.any_horizontal(pl.col(missingness.observed_name(f)).is_null()
                          for f in ("home_ac", "floor")).alias("has_gap"))
    by_gap = (scores.join(gap, on="veteran_id")
                    .group_by("has_gap")
                    .agg(pl.col("p_epistemic_share").mean().alias("share")))
    with_gap = by_gap.filter("has_gap")["share"].item()
    without = by_gap.filter(~pl.col("has_gap"))["share"].item()
    assert with_gap > without, (
        f"veterans with a gap average {with_gap:.5f} epistemic share, veterans without "
        f"{without:.5f} -- the gap buys no uncertainty at all")


# --------------------------------------------------------------------------- #
# What the gap is worth: p_gap_lo / p_gap_hi and the tier that spends a call on them
# --------------------------------------------------------------------------- #

def test_the_gap_range_brackets_the_reported_number(cohort: pl.DataFrame, frames) -> None:
    """`p_gap_lo <= p_mean <= p_gap_hi`, always.

    The three are averaged over the same draws for exactly this reason. Take the extremes at
    the mean coefficients instead and Jensen's gap puts `p_mean` outside its own range at
    these probabilities -- which reads as a bug on the veteran card and silently breaks every
    threshold comparison the tier makes.
    """
    hazards, sites = frames
    hot = hazards.filter(pl.col("hot_day"))["date"].unique().sort().to_list()[:5]
    scores = score_prior.score(cohort, hazards, sites, dates=hot, n_draws=DRAWS, seed=SEED)
    g = scores.filter(pl.col("p_gap_hi").is_not_null())
    assert g.height, "no veteran-day carries a gap range"
    outside = g.filter((pl.col("p_mean") < pl.col("p_gap_lo") - 1e-9)
                       | (pl.col("p_mean") > pl.col("p_gap_hi") + 1e-9))
    assert outside.height == 0, f"p_mean sits outside its own gap range on {outside.height} rows"


def test_the_gap_range_is_null_exactly_when_the_record_is_whole(
        cohort: pl.DataFrame, frames) -> None:
    """Null and zero-width are different answers: "we know this person" versus "knowing more
    would not help". The tier spends a call on neither, but the card says different things."""
    hazards, sites = frames
    days = sorted(hazards["date"].unique().to_list())[:7]
    scores = score_prior.score(cohort, hazards, sites, dates=days, n_draws=DRAWS, seed=SEED)
    gap = cohort.select(
        "veteran_id",
        pl.any_horizontal(pl.col(missingness.observed_name(f)).is_null()
                          for f in ("home_ac", "floor")).alias("has_gap"))
    j = scores.join(gap, on="veteran_id")
    assert j.filter(pl.col("has_gap") & pl.col("p_gap_hi").is_null()).height == 0, \
        "a veteran with a gap was given no gap range"
    assert j.filter(~pl.col("has_gap") & pl.col("p_gap_hi").is_not_null()).height == 0, \
        "a veteran with a complete record was given a gap range"


def test_the_gap_rule_does_not_take_veterans_from_act_now(
        cohort: pl.DataFrame, frames) -> None:
    """The bug this replaces. `act_now` needs the epistemic share *below* `EPISTEMIC_CUT` and
    the old `find_out` needed it above, so the two rules fought over one number and raising
    find-out emptied act-now. The gap rule reads a different column, so it must not move a
    single act-now veteran."""
    hazards, sites = frames
    days = sorted(hazards["date"].unique().to_list())[:7]
    scores = score_prior.score(cohort, hazards, sites, dates=days, n_draws=DRAWS, seed=SEED)

    with_gap = tiers.assign(scores)
    without = tiers.assign(scores.drop("p_gap_lo", "p_gap_hi"))
    n_act = lambda t: t.filter(pl.col("tier") == "act_now").height  # noqa: E731
    assert n_act(with_gap) == n_act(without), (
        f"act-now went from {n_act(without)} to {n_act(with_gap)} veteran-days when the gap "
        f"rule was switched on; it is competing for the same veterans again")


def test_tiers_still_work_on_scores_that_predate_the_gap_columns(
        cohort: pl.DataFrame, frames) -> None:
    """A lane's own fixture, or a parquet written before this landed, must still tier."""
    hazards, sites = frames
    days = sorted(hazards["date"].unique().to_list())[:3]
    scores = score_prior.score(cohort, hazards, sites, dates=days, n_draws=DRAWS, seed=SEED)
    old = tiers.assign(scores.drop("p_gap_lo", "p_gap_hi"))
    assert old.height == scores.select("veteran_id", "date").unique().height
    assert set(old["tier"].unique()) <= set(schema.TIERS)


def test_the_find_out_tier_still_reads_the_share_it_always_did(
        cohort: pl.DataFrame, frames) -> None:
    """Marginalising must not disturb the tiers of veterans whose record is complete.

    This is the regression guard for the decision layer: whatever the find-out share turns
    out to be, a veteran the VA has complete data on must land in the same tier as before.
    """
    hazards, sites = frames
    days = sorted(hazards["date"].unique().to_list())[:7]
    complete = cohort.filter(
        ~pl.any_horizontal(pl.col(missingness.observed_name(f)).is_null()
                           for f in missingness.HIDDEN_FIELDS))
    assert complete.height, "every fixture veteran has a gap"

    hidden = tiers.assign(score_prior.score(complete, hazards, sites, dates=days,
                                            n_draws=DRAWS, seed=SEED))
    revealed = tiers.assign(score_prior.score(_revealed(complete), hazards, sites, dates=days,
                                              n_draws=DRAWS, seed=SEED))
    assert hidden["tier"].to_list() == revealed["tier"].to_list()

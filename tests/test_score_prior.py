"""Rung 0: prior-only scoring. The demo's floor, so its numbers have to be defensible.

The five acceptance checks from docs/PROMPTS.md come first. The ones after them check that
the numbers mean the right thing -- a heat wave raises heat risk, a closed site raises the
treatment gap for the dialysis patient it serves -- because a table can satisfy every
contract and still tell the care team the wrong story.
"""

from __future__ import annotations

import time
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from leeward import schema
from leeward.model import design, priors, score_prior
from leeward.schema import NEEDS
from tables import table


def _dates(n: int, offset: int = 0) -> list:
    return sorted(table("hazards")["date"].unique().to_list())[offset:offset + n]


@pytest.fixture(scope="module")
def scored() -> pl.DataFrame:
    """The full fixture panel over one week."""
    return score_prior.score(table("cohort"), table("hazards"), table("site_status"),
                             dates=_dates(7, offset=10))


# --------------------------------------------------------------------------- #
# The acceptance test from the prompt
# --------------------------------------------------------------------------- #

def test_all_five_needs_for_every_veteran_day(scored: pl.DataFrame) -> None:
    n_vet = table("cohort").height
    assert scored.height == n_vet * 7 * len(NEEDS)
    per = scored.group_by("veteran_id", "date").agg(pl.col("need").sort().alias("needs"))
    assert per.height == n_vet * 7
    assert all(n == sorted(NEEDS) for n in per["needs"].to_list())


def test_every_probability_strictly_inside_zero_one(scored: pl.DataFrame) -> None:
    for col in ("p_mean", "p_lo80", "p_hi80"):
        s = scored[col]
        assert s.null_count() == 0 and s.is_nan().sum() == 0, f"{col} has missing values"
        assert s.min() > 0 and s.max() < 1, f"{col} spans [{s.min()}, {s.max()}]"


def test_intervals_contain_the_mean(scored: pl.DataFrame) -> None:
    bad = scored.filter((pl.col("p_lo80") > pl.col("p_mean")) |
                        (pl.col("p_mean") > pl.col("p_hi80")))
    assert bad.height == 0


def test_epistemic_share_is_a_share(scored: pl.DataFrame) -> None:
    s = scored["p_epistemic_share"]
    assert s.null_count() == 0 and s.is_nan().sum() == 0
    assert s.min() >= 0 and s.max() <= 1


def test_ten_thousand_veterans_by_seven_days_in_under_five_seconds() -> None:
    base = table("cohort")
    big = (base.sample(10_000, with_replacement=True, seed=0)
               .with_columns(veteran_id=pl.format("PERF-{}", pl.int_range(pl.len()))))
    hazards, sites, dates = table("hazards"), table("site_status"), _dates(7)

    t0 = time.perf_counter()
    out = score_prior.score(big, hazards, sites, dates=dates)
    elapsed = time.perf_counter() - t0

    assert out.height == 10_000 * 7 * len(NEEDS)
    assert elapsed < 5.0, f"scoring 10,000 x 7 took {elapsed:.2f}s; the budget is 5s"


# --------------------------------------------------------------------------- #
# The contract and the ladder
# --------------------------------------------------------------------------- #

def test_output_satisfies_the_scores_contract_at_rung_0(scored: pl.DataFrame) -> None:
    schema.validate(scored, "scores")
    assert scored["model_rung"].unique().to_list() == [0]


def test_same_seed_same_numbers() -> None:
    """The same click must produce the same number in rehearsal and on stage."""
    cohort = table("cohort").head(40)
    args = (cohort, table("hazards"), table("site_status"))
    a = score_prior.score(*args, dates=_dates(3))
    b = score_prior.score(*args, dates=_dates(3))
    assert a.equals(b)
    c = score_prior.score(*args, dates=_dates(3), seed=1)
    assert not a["p_hi80"].equals(c["p_hi80"]), "the seed should reach the prior draws"


def test_draws_follow_the_priors() -> None:
    B = priors.draw(4000, seed=0)
    assert B.shape == (4000, len(design.FEATURES), len(NEEDS))
    np.testing.assert_allclose(B.mean(axis=0), priors.mean_matrix(), atol=0.06)

    # A prior that names no need leaves that cell at exactly zero: structure, not shrinkage.
    mask = priors.support_mask()
    assert (B[:, ~mask] == 0).all()

    # prior_scale_multiplier widens the spread and leaves the centre alone.
    wide = priors.draw(4000, seed=0, scale=2.0)
    np.testing.assert_allclose(wide.std(axis=0)[mask], 2 * B.std(axis=0)[mask], rtol=1e-9)


def test_shared_priors_are_one_draw_across_needs() -> None:
    B = priors.draw(50, seed=0)
    for name, prior in priors.PRIORS.items():
        if not isinstance(prior, priors.Prior):
            continue
        term = design.TERM_BY_NAME[name]
        f = design.FEATURES.index(term.feature_names[0])
        cols = [NEEDS.index(n) for n in term.needs]
        assert np.ptp(B[:, f, cols], axis=1).max() == 0, f"{name} should be one parameter"


def test_every_prior_is_documented() -> None:
    """Each prior carries where it came from, so 'why 0.35?' has an answer on stage."""
    for name, prior in priors.PRIORS.items():
        for p in [prior] if isinstance(prior, priors.Prior) else prior.values():
            assert p.source.strip(), f"{name} has a prior with no source"


# --------------------------------------------------------------------------- #
# The numbers mean the right thing
# --------------------------------------------------------------------------- #

def _one_week(cohort: pl.DataFrame, **hazard_overrides) -> tuple[pl.DataFrame, list]:
    hazards = table("hazards")
    dates = _dates(7)
    week = hazards.filter(pl.col("date").is_in(dates))
    week = week.with_columns(**{k: pl.lit(v, dtype=week.schema[k])
                                for k, v in hazard_overrides.items()})
    return score_prior.score(cohort, week, table("site_status").filter(
        pl.col("date").is_in(dates)), dates=dates), dates


def test_a_heat_wave_raises_heat_risk_for_everyone() -> None:
    cohort = table("cohort").head(60)
    calm, _ = _one_week(cohort, heat_index_max_f=74.0, hot_day=False)
    hot, _ = _one_week(cohort, heat_index_max_f=98.0, hot_day=True)
    key = ["veteran_id", "date", "need"]
    both = calm.select(*key, "p_mean").join(hot.select(*key, pl.col("p_mean").alias("hot")),
                                            on=key)
    heat = both.filter(pl.col("need") == "heat")
    assert (heat["hot"] > heat["p_mean"]).all()
    other = both.filter(pl.col("need") == "access_loss")
    assert (other["hot"] >= other["p_mean"]).all(), "heat should never lower another need"


def test_a_closed_site_lands_on_its_dialysis_patient() -> None:
    """The SiteDown term is never cut. It has to reach the veteran it is about."""
    sites = table("site_status")
    down_days = sites.filter((pl.col("facility_id") == "630") & pl.col("site_down"))["date"]
    up_days = sites.filter((pl.col("facility_id") == "630") & ~pl.col("site_down"))["date"]
    if down_days.len() == 0 or up_days.len() == 0:
        pytest.skip("fixture site_status has no 630 closure to test against")
    down, up = down_days.min(), up_days.max()

    # Dialysis at the Manhattan VA, and nothing else that competes for the treatment gap.
    walter = table("cohort").head(1).with_columns(
        veteran_id=pl.lit("WALTER"), facility_id=pl.lit("630"), ckd_dialysis=pl.lit(True),
        med_controlled=pl.lit(False), med_cold_chain=pl.lit(False),
        mail_order_pharmacy=pl.lit(False), powered_equipment=pl.lit("none"))
    out = score_prior.score(walter, table("hazards"), sites, dates=sorted({down, up}))
    tg = out.filter(pl.col("need") == "treatment_gap").sort("date")
    by_day = dict(zip(tg["date"].to_list(), tg["p_mean"].to_list(), strict=True))
    assert by_day[down] > 2 * by_day[up], "closure should more than double the treatment gap"

    row = tg.filter(pl.col("date") == down).row(0, named=True)
    phrase = design.TERM_BY_NAME["theta_sitedown_x_sitedependent"].phrase
    assert row["driver_1"] == phrase, f"top driver on a closure day was {row['driver_1']!r}"
    assert row["driver_1_contrib"] > 0


def test_drivers_are_ordered_and_never_the_intercept(scored: pl.DataFrame) -> None:
    c = scored.select("driver_1_contrib", "driver_2_contrib", "driver_3_contrib")
    a = c.fill_null(0).to_numpy()
    assert (np.abs(a[:, 0]) >= np.abs(a[:, 1])).all() and (np.abs(a[:, 1]) >= np.abs(a[:, 2])).all()
    for i in (1, 2, 3):
        missing_phrase = scored[f"driver_{i}"].is_null()
        missing_value = scored[f"driver_{i}_contrib"].is_null()
        assert (missing_phrase == missing_value).all(), "a driver needs both a phrase and a value"
    phrases = set(scored["driver_1"].drop_nulls().to_list())
    assert design.TERM_BY_NAME["alpha"].phrase not in phrases
    assert phrases <= {t.phrase for t in design.TERMS}


def test_intervals_widen_with_the_prior_scale() -> None:
    cohort = table("cohort").head(40)
    args = (cohort, table("hazards"), table("site_status"))
    narrow = score_prior.score(*args, dates=_dates(2), scale=0.5)
    wide = score_prior.score(*args, dates=_dates(2), scale=2.0)
    w = lambda df: (df["p_hi80"] - df["p_lo80"]).median()  # noqa: E731
    assert w(wide) > w(narrow)


def test_scores_cover_the_requested_window_only() -> None:
    dates = _dates(3, offset=5)
    out = score_prior.score(table("cohort").head(5), table("hazards"), table("site_status"),
                            dates=dates)
    assert sorted(out["date"].unique().to_list()) == dates
    assert dates[0] - timedelta(days=1) not in set(out["date"].to_list())

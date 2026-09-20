"""test_hazard_toy.py -- rung 1's acceptance test (docs/PROMPTS.md §8), written first.

> Fit first on a 200-veteran, 30-day toy cohort; it must complete in under 60 seconds with
> zero post-warmup divergences.

The toy cohort is `tests/tables.py`'s fixture panel, and the outcomes are simulated through
`design.py` from a truth that holds **only the terms rung 1 fits**. That is the honest unit
test of the machinery: when the world is the model, the model has to find the world. What
happens when the world is bigger than the model -- the real `data/truth.json`, whose
interactions rung 1 has no column for -- is a bias, it is large, and it is measured in
`report/recovery.csv` by the eval lane rather than asserted away here.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl
import pytest

from leeward import schema
from leeward.cohort import simulate
from leeward.model import design, fit, hazard, priors, score, score_prior
from leeward.schema import NEEDS
from tables import table

az = pytest.importorskip("arviz")

VETERANS, DAYS = 200, 30
SEED = 0
BUDGET_S = 60.0                  # the acceptance bar from the prompt


# --------------------------------------------------------------------------- #
# The toy world
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def panel() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, list]:
    cohort = table("cohort").head(VETERANS)
    hazards, sites = table("hazards"), table("site_status")
    dates = sorted(hazards["date"].unique().to_list())[:DAYS]
    return cohort, hazards, sites, dates


@pytest.fixture(scope="module")
def truth() -> simulate.Truth:
    """`data/truth.json` with the interaction terms taken out: rung 1's own world."""
    full = simulate.load()
    kept = {k: v for k, v in full.coefficients.items() if k in hazard.MAIN_EFFECTS}
    return dataclasses.replace(full, coefficients=kept, name="main_effects_only")


@pytest.fixture(scope="module")
def outcomes(panel, truth) -> pl.DataFrame:
    cohort, hazards, sites, dates = panel
    # latent=False: no ZIP effect, no frailty. Rung 1 has no term for either, and this test
    # is about the sampler and the design, not about how a marginal model handles what it
    # cannot see. The full fit carries both, and fit.json says so.
    return simulate.simulate(cohort, hazards, sites, dates=dates, truth=truth, seed=SEED,
                             latent=False)


@pytest.fixture(scope="module")
def cells(panel, outcomes) -> hazard.Cells:
    cohort, hazards, sites, dates = panel
    return hazard.cells(cohort, hazards, sites, outcomes, dates=dates)


@pytest.fixture(scope="module")
def fitted(cells) -> fit.Fit:
    return fit.run(cells, chains=fit.CHAINS, warmup=fit.WARMUP, samples=fit.SAMPLES, seed=SEED)


@pytest.fixture(scope="module")
def posterior_path(fitted, tmp_path_factory):
    out = tmp_path_factory.mktemp("posterior")
    nc, js = fit.write(fitted, posterior=out / "posterior.nc", report=out / "fit.json")
    return nc


@pytest.fixture(scope="module")
def draws(posterior_path) -> hazard.Posterior:
    return hazard.read_posterior(posterior_path)


# --------------------------------------------------------------------------- #
# The acceptance test from the prompt
# --------------------------------------------------------------------------- #

def test_two_hundred_veterans_by_thirty_days_fits_inside_the_budget(fitted) -> None:
    assert fitted.cells.n_rows == VETERANS * DAYS
    assert fitted.seconds < BUDGET_S, (
        f"the toy fit took {fitted.seconds:.1f}s; the budget is {BUDGET_S:.0f}s")


def test_no_divergences_after_warmup(fitted) -> None:
    d = fitted.diagnostics
    assert d["divergences"] == 0, f"divergences by chain: {d['divergences_by_chain']}"


def test_rhat_is_under_the_bar_on_every_parameter(fitted) -> None:
    d = fitted.diagnostics
    assert d["rhat_max"] < fit.RHAT_BAR, d["worst_rhat"]
    assert fitted.converged


def test_the_diagnostics_are_the_ones_you_say_on_stage(fitted) -> None:
    """r-hat and divergences for the run, and the rung that produced them."""
    d = fitted.diagnostics
    assert d["model_rung"] == hazard.RUNG == 1
    assert set(d) >= {"rhat_max", "divergences", "ess_bulk_min", "ess_tail_min", "bars",
                      "fitted_terms", "carried_terms", "worst_rhat", "data"}
    assert "rung 1" in fit.summary(fitted) and "divergences" in fit.summary(fitted)


def test_the_toy_clears_the_stricter_published_bar_too(fitted) -> None:
    """SPEC §6.5 asks for r-hat < 1.05. Vehtari et al. (2021) ask for < 1.01 with bulk and
    tail ESS over 400, and that is the number a statistician in the room will want. Both are
    reported; this is the one worth holding the toy fit to."""
    d = fitted.diagnostics
    assert fitted.converged_strict, (d["rhat_max"], d["ess_bulk_min"], d["ess_tail_min"])
    assert d["bars"]["rhat_strict"] == fit.RHAT_BAR_STRICT == 1.01
    assert "clears SPEC's" in fit.summary(fitted) and "not" not in fit._verdict(fitted)


def test_a_run_that_missed_the_bar_is_never_described_as_clearing_it(fitted) -> None:
    """The summary line is read aloud. It said "clears SPEC's r-hat < 1.05" on a run whose
    r-hat was 1.14, because the claim was printed unconditionally."""
    bad = dataclasses.replace(fitted, diagnostics={**fitted.diagnostics, "rhat_max": 1.14})
    assert not bad.converged
    assert "does NOT clear" in fit.summary(bad)
    assert "clears SPEC" not in fit._verdict(bad)


# --------------------------------------------------------------------------- #
# The cells are the panel, exactly
# --------------------------------------------------------------------------- #

def test_the_cell_likelihood_equals_the_bernoulli_panels(panel, outcomes, cells) -> None:
    """The whole reduction rests on this. If it is only approximately true, every posterior
    interval downstream is wrong by an amount nobody can see."""
    cohort, hazards, sites, dates = panel
    B = priors.draw(1, seed=7)[0]
    active = hazard.active_features(hazard.RUNG)

    wide = outcomes.pivot(on="need", index=["veteran_id", "date"], values="y")
    total = 0.0
    for i in range(0, len(dates), 7):
        d = design.build(cohort, hazards, sites, dates=dates[i:i + 7],
                         ref_date=hazards["date"].min())
        rows = pl.DataFrame({"veteran_id": d.veteran_id,
                             "date": pl.Series(d.date).cast(pl.Date)}).join(
            wide, on=["veteran_id", "date"], how="left", maintain_order="left")
        y = rows.select(NEEDS).to_numpy()
        eta = d.X[:, active] @ B[active]
        total += float((y * -np.logaddexp(0, -eta) + (1 - y) * -np.logaddexp(0, eta)).sum())

    assert hazard.log_likelihood(cells, B) == pytest.approx(total, abs=1e-6)


def test_cells_conserve_every_veteran_day_and_every_event(cells, outcomes) -> None:
    assert int(cells.n.sum()) == cells.n_rows == VETERANS * DAYS
    per_need = outcomes.group_by("need").agg(pl.col("y").sum())
    expected = dict(zip(per_need["need"], per_need["y"], strict=True))
    for k, need in enumerate(NEEDS):
        assert int(cells.y[:, k].sum()) == expected[need]
    assert (cells.y <= cells.n[:, None]).all(), "more events than veteran-days in a cell"


def test_the_reduction_is_real(cells) -> None:
    assert cells.X.shape == (cells.X.shape[0], len(cells.names))
    assert cells.X.shape[0] < cells.n_rows
    assert cells.names == tuple(hazard.feature_names(hazard.RUNG))
    assert "theta_heat_x_no_ac" not in " ".join(cells.names), "rung 1 fits no interactions"


def test_two_veterans_who_differ_only_in_an_unfitted_column_are_one_cell(panel) -> None:
    """Grouping is on the active columns, so at rung 1 `home_ac` -- which reaches the design
    only through an interaction -- cannot split a cell."""
    cohort, hazards, sites, dates = panel
    pair = pl.concat([cohort.head(1).with_columns(veteran_id=pl.lit("A"), home_ac=pl.lit(True)),
                      cohort.head(1).with_columns(veteran_id=pl.lit("B"), home_ac=pl.lit(False))])
    day = [d for d in dates if table("hazards").filter(pl.col("date") == d)["hot_day"].any()][0]
    out = simulate.simulate(pair, hazards, sites, dates=[day], latent=False)
    cells = hazard.cells(pair, hazards, sites, out, dates=[day])
    assert cells.X.shape[0] == 1 and int(cells.n[0]) == 2


def test_one_chain_is_refused_before_it_wastes_the_time(cells) -> None:
    """r-hat compares chains. With one there is nothing to compare, and the failure used to
    surface as `max() arg is an empty sequence` several layers down."""
    with pytest.raises(ValueError, match="r-hat"):
        fit.run(cells, chains=1, warmup=10, samples=10)
    with pytest.raises(SystemExit):
        fit.parse_args(["--chains", "1"])


def test_a_missing_outcome_is_an_error_not_a_zero(panel, outcomes) -> None:
    cohort, hazards, sites, dates = panel
    gapped = outcomes.filter(~((pl.col("veteran_id") == cohort["veteran_id"][0])
                               & (pl.col("date") == dates[0])))
    with pytest.raises(ValueError, match="no outcome row"):
        hazard.cells(cohort, hazards, sites, gapped, dates=dates)


# --------------------------------------------------------------------------- #
# The ladder: what was fitted, and what was only carried
# --------------------------------------------------------------------------- #

def test_rung_1_fits_the_main_effects_and_carries_the_interactions(fitted) -> None:
    d = fitted.diagnostics
    assert set(d["fitted_terms"]) == set(hazard.MAIN_EFFECTS)
    assert set(d["carried_terms"]) == set(hazard.INTERACTIONS)
    assert {"alpha", "delta_heat", "psi_sitedown"} <= set(d["fitted_terms"])
    # The terms rung 2 exists for. Named rather than counted, so that adding a term to
    # design.py changes this test only when it changes what the ladder means.
    assert {"theta_sitedown_x_sitedependent", "theta_heat_x_meds",
            "theta_outage_x_equipment"} <= set(d["carried_terms"])


def test_the_ladder_covers_every_design_term_exactly_once() -> None:
    assert set(hazard.LADDER[2]) == set(design.TERM_BY_NAME)
    assert not set(hazard.MAIN_EFFECTS) & set(hazard.INTERACTIONS)


def test_the_same_seed_gives_back_the_same_chain(cells) -> None:
    """"The same click must produce the same number in rehearsal and on stage" (CLAUDE.md).

    JAX on CPU is bitwise reproducible for a fixed version and a fixed input shape, but that
    is a property to measure rather than assume: a posterior that moves between `make fit`
    runs would move every score under it. A short pair of chains is enough to catch it.
    """
    kw = dict(chains=2, warmup=150, samples=150, seed=11)
    draw = lambda f, name: np.asarray(f.idata.posterior[name].values)  # noqa: E731
    a, b, other = fit.run(cells, **kw), fit.run(cells, **kw), fit.run(cells, **{**kw, "seed": 12})
    for name in ("alpha", "delta_heat", "psi_sitedown"):
        np.testing.assert_array_equal(draw(a, name), draw(b, name),
                                      err_msg=f"{name} differs between two identical runs")
    assert not np.array_equal(draw(a, "alpha"), draw(other, "alpha")), (
        "a different seed gave the same chain; the seed is not reaching the sampler")


def test_a_carried_coefficient_comes_back_at_its_prior(draws) -> None:
    """Not zero, and not shrunk: a likelihood that never saw a term leaves it exactly where
    it was. Zeroing it would be a claim of no effect that nothing measured."""
    for name in hazard.INTERACTIONS:
        s = design.TERM_SLICE[name]
        for need, prior in priors.per_need(name).items():
            k = NEEDS.index(need)
            got = draws.B[:, s, k]
            want_mean = np.atleast_1d(np.asarray(prior.mean, dtype=float))
            # Four Monte Carlo standard errors, allowing a 10x autocorrelation penalty on
            # the draw count. These sites are sampled but not observed, so the sampler is
            # drawing from the prior and the error is small; this is not a tight rope.
            tol = 4 * prior.sd / np.sqrt(draws.n_draws / 10)
            assert got.mean(axis=0) == pytest.approx(want_mean, abs=tol), (
                f"{name} ({need}) moved off its prior mean")
            assert got.std(axis=0) == pytest.approx(prior.sd, rel=0.15), (
                f"{name} ({need}) is not as wide as its prior any more")


def test_a_fitted_coefficient_learned_something(draws) -> None:
    """The five intercepts see every veteran-day there is; if the data cannot move those,
    nothing has been fitted at all."""
    s = design.TERM_SLICE["alpha"]
    for need in NEEDS:
        k = NEEDS.index(need)
        prior = priors.per_need("alpha")[need]
        assert draws.B[:, s, k].std() < 0.5 * prior.sd, f"alpha ({need}) did not shrink"


# --------------------------------------------------------------------------- #
# Recovery: the world the toy was given
# --------------------------------------------------------------------------- #

def _interval(draws: hazard.Posterior, p: int, k: int) -> tuple[float, float, float]:
    col = draws.B[:, p, k]
    lo, hi = np.percentile(col, [5, 95])
    return float(lo), float(col.mean()), float(hi)


def test_the_fit_recovers_the_world_it_was_given(draws, truth) -> None:
    """SPEC §11's bar, on the fitted half of the design."""
    B_true = truth.coef_matrix()
    active = hazard.active_features(hazard.RUNG)
    cells_ = [(p, k) for p in range(design.P) for k in range(design.K)
              if design.SUPPORT[p, k] and active[p]]
    missed = []
    for p, k in cells_:
        lo, _, hi = _interval(draws, p, k)
        if not lo <= B_true[p, k] <= hi:
            missed.append(f"{design.FEATURES[p]} ({NEEDS[k]}): {B_true[p, k]:.2f} "
                          f"outside [{lo:.2f}, {hi:.2f}]")
    covered = 1 - len(missed) / len(cells_)
    assert covered >= 0.90, f"{covered:.0%} covered; missed {missed}"


def test_the_intercepts_land_on_the_truth(draws, truth) -> None:
    B_true = truth.coef_matrix()
    p = design.TERM_SLICE["alpha"].start
    for k, need in enumerate(NEEDS):
        lo, mean, hi = _interval(draws, p, k)
        assert lo <= B_true[p, k] <= hi, f"alpha ({need}) {B_true[p, k]:.2f} vs {mean:.2f}"


def test_the_heat_curve_decays_the_way_the_world_did(draws) -> None:
    """Four collinear lags with an RW(0.2) prior (SPEC 6.3) still have to come back ordered:
    today's heat matters more than the heat three days ago."""
    s = design.TERM_SLICE["delta_heat"]
    k = NEEDS.index("heat")
    curve = draws.B[:, s, k].mean(axis=0)
    assert curve[0] > curve[-1] > 0, f"heat lag curve came back {curve.round(2)}"


# --------------------------------------------------------------------------- #
# data/posterior.nc -- the contract the eval lane and the scorer read
# --------------------------------------------------------------------------- #

def test_every_design_term_is_in_the_file(draws) -> None:
    assert draws.B.shape == (fit.CHAINS * fit.SAMPLES, design.P, design.K)
    assert draws.rung == hazard.RUNG
    assert not draws.B[:, ~design.SUPPORT].any(), "a coefficient outside the design's support"
    assert set(draws.fitted) | set(draws.carried) == set(design.TERM_BY_NAME)


def test_the_eval_lane_reads_back_the_same_coefficients(posterior_path) -> None:
    """`eval/recovery.py` has its own reader, written before this module existed. Two
    readers of one contract have to agree, or the report is about a different model."""
    from leeward.eval import recovery as rec

    theirs = rec.posterior_draws(az.from_netcdf(str(posterior_path)))
    np.testing.assert_allclose(theirs, hazard.read_posterior(posterior_path).B,
                               rtol=1e-12, atol=1e-12)


def test_the_evals_diagnostics_survive_the_file(posterior_path, fitted) -> None:
    """`rec.diagnostics` stacks every variable into one array. Give the 35 terms their own
    auto-named dimensions and that stack is 5**35 cells; sharing the `need` dimension keeps
    it at 2,100. This is the test that the dims were named."""
    from leeward.eval import recovery as rec

    rhat, divergences = rec.diagnostics(posterior_path)
    assert np.isfinite(rhat) and rhat < fit.RHAT_BAR
    assert divergences == fitted.diagnostics["divergences"]


def test_a_posterior_that_cannot_name_its_rung_is_refused(posterior_path, tmp_path) -> None:
    idata = az.from_netcdf(str(posterior_path))
    idata.posterior.attrs.pop("model_rung")
    path = tmp_path / "anonymous.nc"
    idata.to_netcdf(str(path))
    with pytest.raises(ValueError, match="which rung"):
        hazard.read_posterior(path)


def test_a_posterior_missing_a_term_is_refused(posterior_path, tmp_path) -> None:
    idata = az.from_netcdf(str(posterior_path))
    kept = az.from_dict(posterior={k: v.values for k, v in idata.posterior.data_vars.items()
                                   if k != "psi_sitedown"})
    kept.posterior.attrs.update(idata.posterior.attrs)
    path = tmp_path / "short.nc"
    kept.to_netcdf(str(path))
    with pytest.raises(ValueError, match="psi_sitedown"):
        hazard.read_posterior(path)


# --------------------------------------------------------------------------- #
# score.py at rung 1
# --------------------------------------------------------------------------- #

def test_score_reads_the_posterior_and_stamps_rung_1(panel, posterior_path) -> None:
    cohort, hazards, sites, dates = panel
    out = score.score(cohort.head(40), hazards, sites, dates=dates[:2], posterior=posterior_path)
    schema.validate(out, "scores")
    assert out["model_rung"].unique().to_list() == [1]
    assert out["p_mean"].min() > 0 and out["p_mean"].max() < 1


def test_score_falls_back_to_the_priors_and_says_rung_0(panel, tmp_path) -> None:
    """A clean clone has no posterior. `make demo` still has to produce a working board,
    and every row has to admit which rung it came from."""
    cohort, hazards, sites, dates = panel
    coef = score.coefficients(posterior=tmp_path / "not_here.nc")
    assert coef.source == "prior" and coef.rung == 0
    out = score.score(cohort.head(20), hazards, sites, dates=dates[:2],
                      posterior=tmp_path / "not_here.nc")
    assert out["model_rung"].unique().to_list() == [0]


def test_scoring_does_not_count_a_carried_term_twice(posterior_path) -> None:
    """The fitted main effects are marginal over the interactions, so the interactions must
    not also be added on top. A carried coefficient reaches the linear predictor as zero."""
    coef = score.coefficients(posterior=posterior_path)
    for name in hazard.INTERACTIONS:
        assert not coef.B[:, design.TERM_SLICE[name], :].any(), f"{name} is in the prediction"
    for name in hazard.MAIN_EFFECTS:
        assert coef.B[:, design.TERM_SLICE[name], :].any(), f"{name} is not"


def test_the_rung_0_path_still_scores_exactly_as_it_did(panel) -> None:
    """`score_prior.py` is now a shim over `score.py`. It has to be the same numbers."""
    cohort, hazards, sites, dates = panel
    small = cohort.head(25)
    a = score_prior.score(small, hazards, sites, dates=dates[:3], seed=1)
    b = score.score(small, hazards, sites, dates=dates[:3], seed=1, posterior=None)
    assert a.equals(b)


def test_a_prior_scale_cannot_be_asked_of_a_posterior(posterior_path) -> None:
    with pytest.raises(ValueError, match="prior_scale"):
        score.coefficients(posterior=posterior_path, scale=2.0)


# --------------------------------------------------------------------------- #
# Cross-lane
# --------------------------------------------------------------------------- #

def test_the_holdout_is_the_one_the_evaluator_uses() -> None:
    """`fit.py` leaves the last 30 days unfitted so `eval/calibration.py` scores a window
    the model never saw. If the two constants drift, the report says held out and means
    nothing of the sort."""
    from leeward.eval import calibration

    assert fit.HOLDOUT_DAYS == calibration.HOLDOUT_DAYS

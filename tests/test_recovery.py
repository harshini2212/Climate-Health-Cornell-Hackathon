"""recovery.py -- do the intervals contain the coefficients the world was generated from?
Written before the module.

`data/truth.json` is the generating truth; `priors.draw` (rung 0) or `data/posterior.nc`
(rung 1 and up) is what the model believes. Recovery is the overlap, per parameter, at 90
percent. SPEC §11's pass bar is 90 percent of parameters covered.

At rung 0 there is no fit, so this measures whether the *priors* bracket the truth. That is
a weaker claim and the module has to say which of the two it made -- a coverage number that
does not name its source is the kind of thing that gets believed on a slide.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

from leeward.api.schemas import RecoveryRow
from leeward.eval import recovery as rec
from leeward.model import design, priors
from leeward.schema import NEEDS

az = pytest.importorskip("arviz")


def _cells() -> list[tuple[int, int]]:
    return [(p, k) for p in range(design.P) for k in range(design.K) if design.SUPPORT[p, k]]


def _draws(values: dict[tuple[int, int], np.ndarray], n: int = 1000) -> np.ndarray:
    """(D, P, K) draws: zero everywhere except the cells `values` names."""
    B = np.zeros((n, design.P, design.K))
    for (p, k), v in values.items():
        B[:, p, k] = v
    return B


# --------------------------------------------------------------------------- #
# The truth side
# --------------------------------------------------------------------------- #

def test_truth_is_read_through_the_shared_design_not_re_keyed() -> None:
    """`design.coef_matrix` is what the simulator used. Reading truth.json any other way
    would let a renamed term become a silent zero on both sides at once."""
    truth = rec.truth_coefficients()
    assert truth.shape == (design.P, design.K)
    assert np.count_nonzero(truth) == design.SUPPORT.sum(), (
        "every cell the design supports should carry a generating coefficient")
    assert not truth[~design.SUPPORT].any(), "truth outside the design's support"


def test_there_is_one_row_per_supported_coefficient_lags_included() -> None:
    tbl = rec.recover(rec.truth_coefficients(), _draws({}))
    assert tbl.height == len(_cells()) == design.SUPPORT.sum()
    # delta_heat is a 4-lag curve acting on one need: four rows, not one.
    heat_lags = tbl.filter(pl.col("parameter").str.starts_with("delta_heat"))
    assert heat_lags.height == 4
    assert heat_lags["parameter"].to_list() == [f"delta_heat[{i}] (heat)" for i in range(4)]


def test_every_parameter_name_is_unique_and_names_its_need() -> None:
    tbl = rec.recover(rec.truth_coefficients(), _draws({}))
    assert tbl["parameter"].n_unique() == tbl.height
    for row in tbl.to_dicts():
        assert row["need"] in NEEDS
        assert row["parameter"].endswith(f"({row['need']})")


# --------------------------------------------------------------------------- #
# The interval and the verdict
# --------------------------------------------------------------------------- #

def test_the_interval_is_the_5th_and_95th_percentile_of_the_draws() -> None:
    p, k = _cells()[3]
    d = np.linspace(0.0, 1.0, 1001)
    tbl = rec.recover(np.zeros((design.P, design.K)), _draws({(p, k): d}, n=1001))
    row = tbl.filter(pl.col("parameter") == rec.parameter_name(p, k)).row(0, named=True)
    assert row["post_mean"] == pytest.approx(0.5)
    assert (row["lo90"], row["hi90"]) == (pytest.approx(0.05), pytest.approx(0.95))


def test_a_truth_inside_the_interval_is_covered_and_one_outside_is_not() -> None:
    (p1, k1), (p2, k2) = _cells()[0], _cells()[1]
    truth = np.zeros((design.P, design.K))
    truth[p1, k1] = 0.5      # inside
    truth[p2, k2] = 9.0      # far outside
    d = np.linspace(0.0, 1.0, 1001)
    tbl = rec.recover(truth, _draws({(p1, k1): d, (p2, k2): d}, n=1001))
    verdict = dict(zip(tbl["parameter"], tbl["covered"], strict=True))
    assert verdict[rec.parameter_name(p1, k1)] is True
    assert verdict[rec.parameter_name(p2, k2)] is False


def test_an_uncovered_parameter_stays_in_the_table() -> None:
    """The row that failed is the one worth reading. Dropping it would be the quiet lie."""
    truth = np.full((design.P, design.K), 9.0)
    tbl = rec.recover(truth, _draws({}))
    assert tbl.height == design.SUPPORT.sum()
    assert not tbl["covered"].any()
    assert rec.coverage(tbl) == 0.0


def test_coverage_is_the_fraction_of_parameters_covered() -> None:
    cells = _cells()
    truth = np.zeros((design.P, design.K))
    for p, k in cells[: len(cells) // 2]:
        truth[p, k] = 9.0
    tbl = rec.recover(truth, _draws({}))
    assert rec.coverage(tbl) == pytest.approx(1 - (len(cells) // 2) / len(cells))


# --------------------------------------------------------------------------- #
# Where the draws come from
# --------------------------------------------------------------------------- #

def test_rung_0_draws_are_the_priors_and_say_so() -> None:
    B, source = rec.coefficient_draws(n_draws=64, seed=0, posterior=None)
    assert B.shape == (64, design.P, design.K)
    assert source == "prior"
    np.testing.assert_allclose(B, priors.draw(64, seed=0), rtol=0, atol=0)


def test_draws_outside_the_design_support_are_exactly_zero() -> None:
    B, _ = rec.coefficient_draws(n_draws=32, seed=1, posterior=None)
    assert not B[:, ~design.SUPPORT].any()


def _idata(B: np.ndarray, n_chains: int = 2):
    """B (D, P, K) -> InferenceData in the canonical (chain, draw, n_features, n_needs) shape."""
    D = B.shape[0]
    posterior = {}
    for term in design.TERMS:
        s = design.TERM_SLICE[term.name]
        block = B[:, s, :].reshape(n_chains, D // n_chains, s.stop - s.start, design.K)
        posterior[term.name] = block
    return az.from_dict(posterior=posterior)


def test_a_posterior_round_trips_back_to_the_same_coefficients() -> None:
    """The rung-1 reader, tested before rung 1 exists: whatever `fit.py` writes under the
    names in priors.py has to come back as the (draws, feature, need) block it went in as."""
    B = priors.draw(8, seed=2)
    back = rec.posterior_draws(_idata(B))
    np.testing.assert_allclose(back, B, rtol=1e-12, atol=1e-12)


def test_a_posterior_variable_of_the_wrong_shape_is_refused() -> None:
    B = priors.draw(8, seed=2)
    idata = _idata(B)
    bad = az.from_dict(posterior={**{k: v.values for k, v in idata.posterior.data_vars.items()},
                                  "beta_copd": np.zeros((2, 4, 7))})
    with pytest.raises(ValueError, match="beta_copd"):
        rec.posterior_draws(bad)


def test_a_posterior_missing_a_term_is_refused_rather_than_zeroed() -> None:
    B = priors.draw(8, seed=2)
    idata = _idata(B)
    kept = {k: v.values for k, v in idata.posterior.data_vars.items() if k != "psi_sitedown"}
    with pytest.raises(ValueError, match="psi_sitedown"):
        rec.posterior_draws(az.from_dict(posterior=kept))


# --------------------------------------------------------------------------- #
# The acceptance bar, and what comes out
# --------------------------------------------------------------------------- #

def test_rung_0_priors_bracket_the_truth_for_at_least_90_percent_of_parameters() -> None:
    """SPEC §11's bar. At rung 0 this is prior coverage, not parameter recovery."""
    B, source = rec.coefficient_draws(n_draws=4000, seed=0, posterior=None)
    tbl = rec.recover(rec.truth_coefficients(), B)
    assert source == "prior"
    assert rec.coverage(tbl) >= 0.90, tbl.filter(~pl.col("covered"))


def test_the_misses_are_named_so_they_can_be_read_off_the_slide() -> None:
    B, _ = rec.coefficient_draws(n_draws=4000, seed=0, posterior=None)
    tbl = rec.recover(rec.truth_coefficients(), B)
    missed = tbl.filter(~pl.col("covered"))
    assert missed.height == tbl.height - int(tbl["covered"].sum())
    for row in missed.to_dicts():
        assert not row["lo90"] <= row["truth"] <= row["hi90"]


def test_recovery_rows_fit_the_report_contract() -> None:
    B, _ = rec.coefficient_draws(n_draws=200, seed=0, posterior=None)
    tbl = rec.recover(rec.truth_coefficients(), B)
    for row in tbl.to_dicts():
        RecoveryRow(parameter=row["parameter"], truth=row["truth"], post_mean=row["post_mean"],
                    lo90=row["lo90"], hi90=row["hi90"], covered=row["covered"])


def test_the_outputs_say_whether_the_interval_was_a_prior_or_a_fit(tmp_path) -> None:
    """`ReportResponse` has nowhere to put this word, so the CSV and the chart carry it.
    A coverage number that does not name its source gets read as a fit."""
    B, source = rec.coefficient_draws(n_draws=200, seed=0, posterior=None)
    tbl = rec.recover(rec.truth_coefficients(), B, source=source)
    csv, html = rec.write_outputs(tbl, tmp_path)
    assert set(pl.read_csv(csv)["source"].to_list()) == {"prior"}
    assert "prior intervals" in html.read_text(encoding="utf-8")


def test_outputs_are_a_tidy_csv_and_an_offline_plotly_chart(tmp_path) -> None:
    B, _ = rec.coefficient_draws(n_draws=200, seed=0, posterior=None)
    tbl = rec.recover(rec.truth_coefficients(), B)
    csv, html = rec.write_outputs(tbl, tmp_path)
    back = pl.read_csv(csv)
    assert back.columns == list(rec.RECOVERY_SCHEMA)
    assert back.height == tbl.height
    page = html.read_text(encoding="utf-8")
    assert "plotly" in page.lower()
    assert not re.search(r"<script[^>]*\bsrc=[\"']https?://", page), (
        "the dot-whisker loads plotly.js from the network; it must open with the wifi off")

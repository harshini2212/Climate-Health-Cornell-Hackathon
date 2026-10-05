"""The heat back-test: the model's heat terms against NYC's real daily heat ED visits.

Hermetic. Every frame is built here or by `tests/tables.py`, seeded, and in memory; nothing
reads `data/` except the one test at the bottom, which reads the committed real series in
`data/reference/` to prove it loads and passes the provenance check.

The observed frames below are *shaped like* the real series and say so with its `source`
tag, because the point of most of these tests is the arithmetic. The provenance tests are
the other half: the back-test must refuse the simulator's outcomes, a `_synthetic` column, or
a frame that cannot say where it came from.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from leeward.eval import backtest as bt
from leeward.ingest.sources import ehdp_heat, heat_syndrome
from leeward.model import design, priors
from leeward.schema import NEEDS
from tables import table

SEED = 20260704
HEAT = NEEDS.index("heat")
B = priors.mean_matrix()


def _season_dates() -> list[date]:
    """May-September, 2017-2021: the shape the real series is published in."""
    out = []
    for y in range(2017, 2022):
        d = date(y, 5, 1)
        while d <= date(y, 9, 30):
            out.append(d)
            d += timedelta(days=1)
    return out


def _temps(seed: int = SEED) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days = _season_dates()
    doy = np.array([d.timetuple().tm_yday for d in days])
    # A summer hump around day 200 plus weather noise: hot spells cross 82F, May does not.
    t = 70 + 16 * np.exp(-((doy - 200) / 45) ** 2) + rng.normal(0, 6, len(days))
    return pl.DataFrame({"date": days, "max_temp_f": np.round(t)})


def _observed(visits: np.ndarray, temps: pl.DataFrame) -> pl.DataFrame:
    return temps.with_columns(heat_ed_visits=pl.Series(visits.astype(np.int64)),
                              file=pl.lit(heat_syndrome.FILES[0]),
                              source=pl.lit(heat_syndrome.SOURCE))


@pytest.fixture(scope="module")
def cohort() -> pl.DataFrame:
    return table("cohort")


@pytest.fixture(scope="module")
def temps() -> pl.DataFrame:
    return _temps()


@pytest.fixture(scope="module")
def daily(cohort, temps) -> pl.DataFrame:
    return bt.predict(cohort, temps, B)


# --------------------------------------------------------------------------- #
# Provenance: the back-test refuses synthetic outcomes
# --------------------------------------------------------------------------- #

def test_refuses_the_simulators_outcome_table() -> None:
    with pytest.raises(bt.SyntheticOutcomes, match="veteran"):
        bt.require_real(table("outcomes"))


def test_refuses_a_frame_carrying_a_synthetic_flag(temps) -> None:
    obs = _observed(np.ones(temps.height), temps).with_columns(
        heat_ed_visits_synthetic=pl.lit(True))
    with pytest.raises(bt.SyntheticOutcomes, match="_synthetic"):
        bt.require_real(obs)


def test_refuses_a_frame_from_any_other_source(temps) -> None:
    obs = _observed(np.ones(temps.height), temps).with_columns(source=pl.lit("simulate.py"))
    with pytest.raises(bt.SyntheticOutcomes, match="simulate.py"):
        bt.require_real(obs)


def test_refuses_a_frame_that_cannot_say_where_it_came_from(temps) -> None:
    obs = _observed(np.ones(temps.height), temps).drop("source")
    with pytest.raises(bt.SyntheticOutcomes, match="source"):
        bt.require_real(obs)


def test_refuses_rows_from_a_file_nychealth_did_not_publish(temps) -> None:
    obs = _observed(np.ones(temps.height), temps).with_columns(file=pl.lit("simulated.csv"))
    with pytest.raises(bt.SyntheticOutcomes, match="simulated.csv"):
        bt.require_real(obs)


def test_the_loader_refuses_a_file_whose_bytes_changed_since_the_fetch(temps, tmp_path) -> None:
    """A frame can claim any `source`; the committed file must still hash to the manifest."""
    fake = tmp_path / bt.OBSERVED.name
    _observed(np.ones(temps.height), temps).write_parquet(fake)
    with pytest.raises(bt.SyntheticOutcomes, match="sha256"):
        bt.load_observed(fake)


def test_the_run_itself_checks_provenance_not_just_the_loader(cohort) -> None:
    with pytest.raises(bt.SyntheticOutcomes):
        bt.backtest(table("outcomes"), cohort, B, rung=0, detail="t", population=8e6)


# --------------------------------------------------------------------------- #
# The prediction is exactly the model's heat column
# --------------------------------------------------------------------------- #

def test_prediction_equals_the_full_design_matrix_on_every_day(cohort) -> None:
    """The fast path splits the heat log-odds into person + hot-day + lag parts. Check that
    against `design.build` over every veteran-day, so the split cannot drift from the model."""
    temps = pl.DataFrame({"date": [date(2019, 7, 1) + timedelta(days=i) for i in range(8)],
                          "max_temp_f": [70.0, 81.0, 82.0, 95.0, 101.0, 88.0, 75.0, 120.0]})
    fast = bt.predict(cohort, temps, B)["intensity"].to_numpy()

    d = design.build(cohort, bt.citywide_hazards(cohort, temps), bt.calm_sites(cohort, temps))
    p = 1 / (1 + np.exp(-(d.X @ B[:, HEAT])))
    slow = pl.DataFrame({"date": d.date, "p": p}).group_by("date").agg(pl.col("p").mean())
    np.testing.assert_allclose(fast, slow.sort("date")["p"].to_numpy(), rtol=1e-12)


def test_one_hot_day_echoes_for_exactly_four_days_with_the_prior_lag_curve(cohort) -> None:
    temps = pl.DataFrame({"date": [date(2019, 7, 1) + timedelta(days=i) for i in range(7)],
                          "max_temp_f": [70.0, 102.0, 70.0, 70.0, 70.0, 70.0, 70.0]})
    lag = bt.predict(cohort, temps, B)["lag_logit"].to_list()
    delta = priors.PRIORS["delta_heat"]["heat"].mean
    x = (102 - design.HEAT_HINGE_F) / design.HEAT_UNIT_F
    assert lag == pytest.approx([0.0, delta[0] * x, delta[1] * x, delta[2] * x, delta[3] * x,
                                 0.0, 0.0])


def test_intensity_is_a_probability_and_rises_with_heat(daily) -> None:
    p = daily["intensity"].to_numpy()
    assert ((p > 0) & (p < 1)).all()
    hot = daily.filter(pl.col("max_temp_f") >= 95)["intensity"].mean()
    cool = daily.filter(pl.col("max_temp_f") < 75)["intensity"].mean()
    assert hot > cool


# --------------------------------------------------------------------------- #
# The metrics
# --------------------------------------------------------------------------- #

def test_spearman_matches_a_hand_computed_value_with_ties() -> None:
    assert bt.spearman(np.array([1, 2, 3, 4.0]), np.array([1, 3, 2, 4.0])) == pytest.approx(0.8)
    # Ties take the average rank: [1, 2.5, 2.5, 4] against [1, 2, 3, 4].
    assert bt.spearman(np.array([1, 2, 2, 3.0]), np.array([1, 2, 3, 4.0])) == pytest.approx(
        np.corrcoef([1, 2.5, 2.5, 4], [1, 2, 3, 4])[0, 1])


def test_a_series_driven_by_the_model_scores_high_and_its_shuffle_scores_nothing(
        cohort, temps, daily) -> None:
    rng = np.random.default_rng(SEED)
    lam = 40 * daily["intensity"].to_numpy() / daily["intensity"].mean()
    driven = _observed(rng.poisson(lam), temps)
    shuffled = _observed(rng.permutation(driven["heat_ed_visits"].to_numpy()), temps)

    good, _ = bt.backtest(driven, cohort, B, rung=0, detail="t", population=8e6)
    null, _ = bt.backtest(shuffled, cohort, B, rung=0, detail="t", population=8e6)
    assert good["rank_correlation"]["spearman"] > 0.7
    assert abs(null["rank_correlation"]["spearman"]) < 0.15


def test_the_observed_lag_response_finds_a_two_day_delay(cohort, temps, daily) -> None:
    rng = np.random.default_rng(SEED)
    x2 = daily["heat_x2"].to_numpy()
    obs = _observed(rng.poisson(np.exp(1.0 + 1.5 * x2)), temps)
    report, _ = bt.backtest(obs, cohort, B, rung=0, detail="t", population=8e6)
    lag = report["lag_structure"]
    coef = lag["observed_poisson_coef"]
    assert int(np.argmax(coef)) == 2
    assert coef[2] == pytest.approx(1.5, abs=0.15)
    assert int(np.argmax(lag["observed_vs_heat_x"])) == 2
    assert len(lag["predicted_vs_observed_xcorr"]) == design.HEAT_LAGS


def test_calibration_in_the_large_is_predicted_rate_over_observed_rate(
        cohort, temps, daily) -> None:
    obs = _observed(np.full(temps.height, 8), temps)
    report, out = bt.backtest(obs, cohort, B, rung=0, detail="t", population=8e6)
    citl = report["calibration_in_the_large"]
    assert citl["observed_rate_per_person_day"] == pytest.approx(8 / 8e6)
    assert citl["predicted_rate_per_person_day"] == pytest.approx(out["intensity"].mean())
    assert citl["ratio"] == pytest.approx(out["intensity"].mean() / (8 / 8e6))


def test_scores_only_june_to_august_2017_to_2021_but_lags_reach_back_into_may(
        cohort, temps) -> None:
    obs = _observed(np.ones(temps.height), temps)
    report, out = bt.backtest(obs, cohort, B, rung=0, detail="t", population=8e6)
    assert set(out["date"].dt.month()) == {6, 7, 8}
    assert set(out["date"].dt.year()) == set(range(2017, 2022))
    assert report["n_days"] == out.height == 5 * 92
    # 1 June's lag-3 is 29 May: read from the May rows, not zero-filled.
    first = out.filter(pl.col("date") == date(2017, 6, 1)).row(0, named=True)
    may29 = temps.filter(pl.col("date") == date(2017, 5, 29))["max_temp_f"][0]
    want = min(max((may29 - design.HEAT_HINGE_F) / design.HEAT_UNIT_F, 0), design.HEAT_CAP)
    assert first["heat_x3"] == pytest.approx(want)


def test_the_rung_is_chosen_explicitly_never_by_whether_a_posterior_exists(tmp_path) -> None:
    with pytest.raises(SystemExit):
        bt.main(["--out", str(tmp_path)])                     # neither --prior nor --fitted
    with pytest.raises(SystemExit):
        bt.main(["--fitted", "--posterior", str(tmp_path / "none.nc"), "--out", str(tmp_path)])


def test_a_negative_lag_weight_gets_no_shape_and_calm_days_are_counted(cohort, temps) -> None:
    rng = np.random.default_rng(SEED)
    x = bt.predict(cohort, temps, B)
    # Visits fall the day after heat: the fitted lag-1 weight is negative.
    obs = _observed(rng.poisson(np.exp(1.0 + 1.0 * x["heat_x0"] - 1.0 * x["heat_x1"])), temps)
    report, out = bt.backtest(obs, cohort, B, rung=0, detail="t", population=8e6)
    assert report["lag_structure"]["observed_poisson_coef"][1] < 0
    assert report["lag_structure"]["observed_shape"] is None
    calm = out.filter(pl.all_horizontal([pl.col(f"heat_x{i}") == 0 for i in range(4)]))
    assert report["n_days_at_baseline"] == calm.height > 0


def test_refuses_a_series_with_a_hole_in_summer(cohort, temps) -> None:
    obs = _observed(np.ones(temps.height), temps).filter(pl.col("date") != date(2019, 7, 4))
    with pytest.raises(ValueError, match="2019-07-04"):
        bt.backtest(obs, cohort, B, rung=0, detail="t", population=8e6)


def test_same_inputs_same_report(cohort, temps, daily) -> None:
    rng = np.random.default_rng(SEED)
    obs = _observed(rng.poisson(5, temps.height), temps)
    a, _ = bt.backtest(obs, cohort, B, rung=0, detail="t", population=8e6)
    b, _ = bt.backtest(obs, cohort, B, rung=0, detail="t", population=8e6)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_spatial_check_ranks_districts_when_given_ehdp_rates(cohort, temps) -> None:
    obs = _observed(np.ones(temps.height), temps)
    zips = sorted(cohort["geo_id"].unique().to_list())
    geo = pl.DataFrame({"geo_id": zips, "cd": [str(101 + i % 6) for i in range(len(zips))]})
    rates = pl.DataFrame({
        "indicator_id": bt.SPATIAL_INDICATOR, "outcome": "heat_ed_visits",
        "measure": "Estimated annual rate", "geo_type": "CD",
        "geo_id": [str(101 + i) for i in range(6)], "geo_name": [f"CD{i}" for i in range(6)],
        "time_period": bt.SPATIAL_PERIOD, "value": [1.0, 2.0, None, 4.0, 5.0, 6.0],
        "source": ehdp_heat.SOURCE,
    })
    report, _ = bt.backtest(obs, cohort, B, rung=0, detail="t", population=8e6,
                            ehdp=rates, geo=geo, min_veterans=1)
    sp = report["spatial"]
    assert sp["n_neighbourhoods"] == 5, "a suppressed district is dropped, not read as zero"
    assert -1 <= sp["spearman"] <= 1


def test_spatial_check_refuses_rates_from_anywhere_but_ehdp(cohort, temps) -> None:
    obs = _observed(np.ones(temps.height), temps)
    geo = pl.DataFrame({"geo_id": cohort["geo_id"].unique(), "cd": "101"})
    rates = pl.DataFrame({"indicator_id": [2443], "outcome": ["heat_ed_visits"],
                          "measure": ["Estimated annual rate"], "geo_type": ["CD"],
                          "geo_id": ["101"], "geo_name": ["x"], "time_period": ["2018-22"],
                          "value": [1.0], "source": ["simulate.py"]})
    with pytest.raises(bt.SyntheticOutcomes, match="simulate.py"):
        bt.backtest(obs, cohort, B, rung=0, detail="t", population=8e6, ehdp=rates, geo=geo)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def test_writes_json_and_one_chart_that_opens_offline(cohort, temps, tmp_path) -> None:
    obs = _observed(np.ones(temps.height), temps)
    report, out = bt.backtest(obs, cohort, B, rung=0, detail="t", population=8e6)
    js, html = bt.write(report, out, tmp_path)
    assert json.loads(js.read_text())["n_days"] == 460
    page = html.read_text()
    assert "<script src=\"http" not in page, "the chart must inline plotly.js, not fetch it"


# --------------------------------------------------------------------------- #
# The committed real series (data/reference is committed input, so this is stable)
# --------------------------------------------------------------------------- #

def test_the_committed_series_is_real_and_covers_every_summer_day() -> None:
    obs = bt.load_observed()
    assert obs.height == 765
    assert obs["date"].min() == date(2017, 5, 1) and obs["date"].max() == date(2021, 9, 30)
    summer = obs.filter(pl.col("date").dt.month().is_in([6, 7, 8]))
    assert summer.height == 460

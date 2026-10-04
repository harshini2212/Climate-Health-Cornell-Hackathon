"""The shared feature mapping. The simulator and every model rung read the world through it,
so a wrong join here is wrong everywhere at once -- these check that each feature means
what its name says, on hand-set hazards where the right answer is obvious.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from leeward.model import design
from leeward.schema import NEEDS
from tables import table

D0 = date(2026, 7, 1)
DAYS = [D0 + timedelta(days=i) for i in range(6)]


def _vet(**overrides) -> pl.DataFrame:
    """One fixture veteran, with the fields a test cares about set by hand."""
    base = table("cohort").head(1)
    return base.with_columns(**{k: pl.lit(v, dtype=base.schema[k]) for k, v in overrides.items()})


def _calm_hazards(modzctas: list[str]) -> pl.DataFrame:
    """72F, clean air, dry, powered, mail running: every hazard feature should read zero."""
    n = len(modzctas) * len(DAYS)
    return pl.DataFrame({
        "modzcta": [z for z in modzctas for _ in DAYS],
        "date": pl.Series([d for _ in modzctas for d in DAYS], dtype=pl.Date),
        "heat_index_max_f": [72.0] * n,
        "hot_day": [False] * n,
        "heat_alert": [False] * n,
        "pm25": [8.0] * n,
        "smoke_alert": [False] * n,
        "flood_watch": [False] * n,
        "flood_warning": [False] * n,
        "flash_flood_emergency": [False] * n,
        "surge_ft": [0.0] * n,
        "evac_zone_ordered": pl.Series([0] * n, dtype=pl.Int32),
        "floodnet_trip": [False] * n,
        "stormwater_flooded_frac": [0.1] * n,
        "outage_frac": [0.0] * n,
        "mail_delivery_disrupted": [False] * n,
    })


def _sites(facilities: list[str], down: dict[str, set[date]] | None = None) -> pl.DataFrame:
    down = down or {}
    return pl.DataFrame({
        "facility_id": [f for f in facilities for _ in DAYS],
        "date": pl.Series([d for _ in facilities for d in DAYS], dtype=pl.Date),
        "site_down": [d in down.get(f, set()) for f in facilities for d in DAYS],
        "evac_zone": pl.Series([1] * (len(facilities) * len(DAYS)), dtype=pl.Int32),
        "site_dependent_services": [True] * (len(facilities) * len(DAYS)),
    })


def _set(h: pl.DataFrame, day: date, **values) -> pl.DataFrame:
    on = pl.col("date") == day
    return h.with_columns(**{k: pl.when(on).then(pl.lit(v)).otherwise(pl.col(k)).alias(k)
                             for k, v in values.items()})


def _feature(d, term: str, lag: int | None = None) -> np.ndarray:
    name = term if lag is None else f"{term}[{lag}]"
    return d.X[:, design.FEATURES.index(name)]


def _build(cohort, hazards, sites, **kw):
    return design.build(cohort, hazards, sites, **kw)


# --------------------------------------------------------------------------- #
# Shape and order
# --------------------------------------------------------------------------- #

def test_one_row_per_veteran_day_in_cohort_then_date_order() -> None:
    cohort = table("cohort").head(7)
    hazards, sites = table("hazards"), table("site_status")
    dates = sorted(hazards["date"].unique().to_list())[:3]
    d = _build(cohort, hazards, sites, dates=dates)

    assert d.X.shape == (7 * 3, len(design.FEATURES))
    assert np.isfinite(d.X).all()
    assert list(d.veteran_id) == [v for v in cohort["veteran_id"] for _ in dates]
    assert [str(x) for x in d.date] == [str(x) for _ in range(7) for x in dates]


def test_calm_day_zeroes_every_hazard_feature() -> None:
    vet = _vet()
    d = _build(vet, _calm_hazards(vet["modzcta"].to_list()),
               _sites(vet["facility_id"].to_list()))
    for term in design.TERMS:
        if term.kind != "hazard":
            continue
        cols = [design.FEATURES.index(f) for f in term.feature_names]
        assert (d.X[:, cols] == 0).all(), f"{term.name} is non-zero on a calm day"


def test_missing_hazard_row_fails_loudly() -> None:
    """A ZIP with no hazard row must not silently read as 'no hazard'."""
    vet = _vet()
    hz = _calm_hazards(vet["modzcta"].to_list()).filter(pl.col("date") != DAYS[2])
    with pytest.raises(ValueError, match="hazard"):
        _build(vet, hz, _sites(vet["facility_id"].to_list()), dates=DAYS)


def test_missing_site_status_fails_loudly() -> None:
    vet = _vet()
    with pytest.raises(ValueError, match="site_status"):
        _build(vet, _calm_hazards(vet["modzcta"].to_list()), _sites(["not-a-station"]))


# --------------------------------------------------------------------------- #
# Each feature means what its name says
# --------------------------------------------------------------------------- #

def test_heat_hinges_at_82F_and_lags_carry_forward() -> None:
    vet = _vet(med_thermoreg_score=2.0)
    hz = _set(_calm_hazards(vet["modzcta"].to_list()), DAYS[2],
              heat_index_max_f=92.0, hot_day=True)
    d = _build(vet, hz, _sites(vet["facility_id"].to_list()))

    lag0 = _feature(d, "delta_heat", 0)
    assert lag0[2] == pytest.approx(1.0), "92F is one 10-degree unit above NYC's 82F hinge"
    assert lag0[[0, 1, 3, 4, 5]].tolist() == [0, 0, 0, 0, 0]
    for lag in range(1, design.HEAT_LAGS):
        f = _feature(d, "delta_heat", lag)
        assert f[2 + lag] == pytest.approx(1.0), f"lag {lag} should see day 2 on day {2 + lag}"
        assert f.sum() == pytest.approx(1.0)

    meds = _feature(d, "theta_heat_x_meds")
    assert meds.tolist() == [0, 0, 2.0, 0, 0, 0], "heat x meds fires on the hot day, per score unit"


def test_first_day_lags_read_zero_not_garbage() -> None:
    vet = _vet()
    hz = _set(_calm_hazards(vet["modzcta"].to_list()), DAYS[0],
              heat_index_max_f=102.0, hot_day=True)
    d = _build(vet, hz, _sites(vet["facility_id"].to_list()))
    assert _feature(d, "delta_heat", 0)[0] == pytest.approx(2.0)
    for lag in range(1, design.HEAT_LAGS):
        assert _feature(d, "delta_heat", lag)[0] == 0


def test_smoke_above_the_epa_standard_only() -> None:
    vet = _vet(pact_presumptive=True)
    hz = _calm_hazards(vet["modzcta"].to_list())
    hz = _set(hz, DAYS[1], pm25=30.0)            # bad air, under the 24-hour standard
    hz = _set(hz, DAYS[3], pm25=203.5)           # the real 7 June 2023 Queens peak
    d = _build(vet, hz, _sites(vet["facility_id"].to_list()))
    lag0 = _feature(d, "eps_pm25", 0)
    assert lag0[1] == 0
    assert lag0[3] == pytest.approx((203.5 - design.PM25_STANDARD) / design.PM25_UNIT)
    assert _feature(d, "theta_smoke_x_pact").tolist() == [0, 0, 0, 1, 0, 0]


def test_site_down_reaches_only_site_dependent_veterans_at_that_site() -> None:
    dialysis = _vet(veteran_id="V-DIAL", facility_id="630", ckd_dialysis=True,
                    on_methadone_otp=False, active_cancer_tx=False, med_controlled=False)
    elsewhere = _vet(veteran_id="V-ELSE", facility_id="630A4", ckd_dialysis=True,
                     on_methadone_otp=False, active_cancer_tx=False, med_controlled=False)
    plain = _vet(veteran_id="V-PLAIN", facility_id="630", ckd_dialysis=False,
                 on_methadone_otp=False, active_cancer_tx=False, med_controlled=False)
    cohort = pl.concat([dialysis, elsewhere, plain])
    hz = _calm_hazards(cohort["modzcta"].unique().to_list())
    sites = _sites(["630", "630A4"], down={"630": {DAYS[3], DAYS[4]}})
    d = _build(cohort, hz, sites)

    sd = _feature(d, "theta_sitedown_x_sitedependent").reshape(3, len(DAYS))
    assert sd[0].tolist() == [0, 0, 0, 1, 1, 0], "dialysis at 630 while 630 is down"
    assert sd[1].sum() == 0, "630A4 stayed open"
    assert sd[2].sum() == 0, "not site-dependent"
    down = _feature(d, "psi_sitedown").reshape(3, len(DAYS))
    assert down[2].tolist() == [0, 0, 0, 1, 1, 0], "every veteran at 630 feels the closure"


def test_site_down_needs_the_site_to_offer_the_service() -> None:
    vet = _vet(facility_id="630", ckd_dialysis=True)
    sites = _sites(["630"], down={"630": {DAYS[1]}}).with_columns(
        site_dependent_services=pl.lit(False))
    d = _build(vet, _calm_hazards(vet["modzcta"].to_list()), sites)
    assert _feature(d, "theta_sitedown_x_sitedependent").sum() == 0


def test_evacuation_order_hits_only_ordered_zones() -> None:
    zones = {"V-Z1": 1, "V-Z3": 3, "V-Z0": 0}
    cohort = pl.concat([_vet(veteran_id=v, evac_zone=z) for v, z in zones.items()])
    hz = _set(_calm_hazards(cohort["modzcta"].unique().to_list()), DAYS[2], evac_zone_ordered=2)
    d = _build(cohort, hz, _sites(cohort["facility_id"].unique().to_list()))
    ev = _feature(d, "zeta_evac").reshape(3, len(DAYS))[:, 2]
    assert ev.tolist() == [1, 0, 0], "zone 1 is inside an order for zones 1-2; zone 3 and 0 are not"


def test_mail_disruption_needs_mail_order_and_a_short_supply() -> None:
    short = _vet(veteran_id="V-SHORT", mail_order_pharmacy=True, days_supply_remaining=3)
    stocked = _vet(veteran_id="V-STOCK", mail_order_pharmacy=True, days_supply_remaining=60)
    retail = _vet(veteran_id="V-RETAIL", mail_order_pharmacy=False, days_supply_remaining=3)
    cohort = pl.concat([short, stocked, retail])
    hz = _calm_hazards(cohort["modzcta"].unique().to_list()).with_columns(
        mail_delivery_disrupted=pl.lit(True))
    d = _build(cohort, hz, _sites(cohort["facility_id"].unique().to_list()))
    m = _feature(d, "theta_mail_x_supply").reshape(3, len(DAYS))
    assert m[0, 0] == 1 and m[1, 0] == 0 and m[2, 0] == 0


def test_supply_counts_down_from_the_reference_date_and_refills() -> None:
    vet = _vet(mail_order_pharmacy=True, days_supply_remaining=12)
    hz = _calm_hazards(vet["modzcta"].to_list()).with_columns(mail_delivery_disrupted=pl.lit(True))
    d = _build(vet, hz, _sites(vet["facility_id"].to_list()), ref_date=D0)
    short = _feature(d, "theta_mail_x_supply").tolist()
    # 12 days left on D0: short once 7 or fewer remain, i.e. from day 5.
    assert short == [0, 0, 0, 0, 0, 1]

    late = _build(vet, hz, _sites(vet["facility_id"].to_list()), ref_date=D0 - timedelta(days=20))
    # 20 days before D0 there were 12 left; the 90-day mail fill has since come round.
    assert _feature(late, "theta_mail_x_supply").sum() == 0


# --------------------------------------------------------------------------- #
# Coefficients and the linear predictor
# --------------------------------------------------------------------------- #

def test_coef_matrix_places_named_params() -> None:
    B = design.coef_matrix({
        "alpha": {"heat": -6.0, "breathing": -5.5},
        "delta_heat": {"heat": [0.9, 0.7, 0.4, 0.15]},
        "theta_no_caregiver_x_hazard": 0.5,
    })
    assert B.shape == (len(design.FEATURES), len(NEEDS))
    k = NEEDS.index("heat")
    assert B[design.FEATURES.index("alpha"), k] == -6.0
    assert B[design.FEATURES.index("delta_heat[2]"), k] == 0.4
    nc = design.FEATURES.index("theta_no_caregiver_x_hazard")
    term = design.TERM_BY_NAME["theta_no_caregiver_x_hazard"]
    for need in NEEDS:
        want = 0.5 if need in term.needs else 0.0
        assert B[nc, NEEDS.index(need)] == want


@pytest.mark.parametrize("bad", [
    {"theta_heat_x_mets": {"heat": 0.7}},              # typo in the term
    {"theta_heat_x_meds": {"mental": 0.7}},            # a need this term does not touch
    {"delta_heat": {"heat": [0.9, 0.7]}},              # wrong number of lags
    {"alpha": {"heat_stroke": -6.0}},                  # not a need
])
def test_coef_matrix_rejects_typos(bad) -> None:
    """A truth.json typo must fail at load, not produce a simulator that ignores the term."""
    with pytest.raises(ValueError):
        design.coef_matrix(bad)


def test_linear_predictor_and_contributions_agree() -> None:
    cohort = table("cohort").head(20)
    hazards, sites = table("hazards"), table("site_status")
    dates = sorted(hazards["date"].unique().to_list())[10:14]
    d = _build(cohort, hazards, sites, dates=dates)

    rng = np.random.default_rng(0)
    draws = rng.normal(0, 0.5, (3, len(design.FEATURES), len(NEEDS)))
    eta = design.linear_predictor(d.X, draws)                     # (N, K, D)
    assert eta.shape == (d.X.shape[0], len(NEEDS), 3)
    for i in range(3):
        np.testing.assert_allclose(eta[:, :, i], design.linear_predictor(d.X, draws[i]))

    contrib = design.term_contributions(d.X, draws[0])            # (N, T, K)
    assert contrib.shape == (d.X.shape[0], len(design.TERMS), len(NEEDS))
    np.testing.assert_allclose(contrib.sum(axis=1), eta[:, :, 0], atol=1e-9)

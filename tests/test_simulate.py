"""Acceptance tests for `leeward/cohort/simulate.py` -- the generative truth.

Written before the module. The simulator is the only thing in the build that knows the right
answer, so if it is wrong then every recovery, calibration and fairness number downstream is
wrong *and agrees with itself*. These check the two properties that would catch that: the
intercept means what it says on a calm day, and a closed site moves exactly the people it
should move, by roughly the amount `truth.json` says.

`sigmoid(alpha_k)` is the rate for a veteran with **no risk factors** on a calm day, so the
intercept test uses veterans with every person feature switched off and the latent ZIP and
frailty offsets disabled. Averaging a real cohort instead would mix in every beta the
simulator deliberately plants -- 1.5 to 2.5x higher, by construction -- which is checked
separately against SPEC 5.5's 0.2-2%/day off-event band.

Frames come from `tests/tables.py`, never from `data/`: these numbers must not depend on
which make target ran last.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from leeward import schema
from leeward.cohort import simulate
from leeward.model import design
from leeward.schema import NEEDS
from tables import table

D0 = date(2026, 7, 1)

#: The six terms that cannot fire until `cohort/medications.py` lands: every feature they
#: multiply is a medication column, and `cohort/build.py` leaves those at zero today.
DORMANT_MED_TERMS = ["theta_heat_x_meds", "theta_heat_x_acb", "theta_heat_x_raas_diuretic",
                     "theta_heat_x_renal_triple", "theta_outage_x_cold_chain",
                     "theta_sitedown_x_controlled"]

#: `theta_mail_x_supply` is planted with them but is *already live*: build.py draws
#: `mail_order_pharmacy` and `days_supply_remaining` today, so it fires on a mail disruption
#: without waiting for a prescription.
PLANTED_MED_TERMS = [*DORMANT_MED_TERMS, "theta_mail_x_supply"]

#: The medication columns those six terms multiply. Zeroing them is today's cohort.
MED_COLUMNS = dict(med_thermoreg_score=0.0, acb_score=0, med_combo_raas_diuretic=False,
                   med_renal_triple=False, med_cold_chain=False, med_controlled=False)

#: Every person feature off: under 65, no conditions, a caregiver, money, no medication.
#: What is left is the intercept.
NEUTRAL = dict(
    age=40, copd=False, asthma=False, chf=False, diabetes=False, ckd_dialysis=False,
    active_cancer_tx=False, ptsd=False, depression=False, pact_presumptive=False,
    mobility_impaired=False, caregiver="informal_coresident", low_assets=False,
    home_ac=True, floor="upper", powered_equipment="none", on_methadone_otp=False,
    evac_zone=0, mail_order_pharmacy=False, days_supply_remaining=90, **MED_COLUMNS,
)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture(scope="module")
def truth() -> simulate.Truth:
    """The committed `data/truth.json` -- the artifact, not a copy of it."""
    return simulate.load()


def _days(n: int, start: date = D0) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _vet(**overrides) -> pl.DataFrame:
    """One veteran, with the fields a test cares about set by hand."""
    base = table("cohort").head(1)
    return base.with_columns(**{k: pl.lit(v, dtype=base.schema[k]) for k, v in overrides.items()})


def _many(vet: pl.DataFrame, n: int, prefix: str) -> pl.DataFrame:
    """`n` copies of one veteran with distinct ids -- an iid panel to measure a rate on."""
    return (vet.join(pl.DataFrame({"_r": pl.arange(0, n, eager=True)}), how="cross")
               .drop("_r")
               .with_columns(veteran_id=pl.format(prefix + "-{}", pl.int_range(pl.len()))))


def _calm(modzctas: list[str], days: list[date], **overrides) -> pl.DataFrame:
    """72F, clean air, dry, powered, mail running: every hazard feature reads zero."""
    n = len(modzctas) * len(days)
    hz = pl.DataFrame({
        "modzcta": [z for z in modzctas for _ in days],
        "date": pl.Series([d for _ in modzctas for d in days], dtype=pl.Date),
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
    return hz.with_columns(**{k: pl.lit(v) for k, v in overrides.items()})


def _sites(facilities: list[str], days: list[date],
           down: dict[str, set[date]] | None = None) -> pl.DataFrame:
    down = down or {}
    n = len(facilities) * len(days)
    return pl.DataFrame({
        "facility_id": [f for f in facilities for _ in days],
        "date": pl.Series([d for _ in facilities for d in days], dtype=pl.Date),
        "site_down": [d in down.get(f, set()) for f in facilities for d in days],
        "evac_zone": pl.Series([1] * n, dtype=pl.Int32),
        "site_dependent_services": [True] * n,
    })


def _world(cohort: pl.DataFrame, days: list[date], **hazard_overrides):
    return (_calm(cohort["modzcta"].unique().to_list(), days, **hazard_overrides),
            _sites(cohort["facility_id"].unique().to_list(), days))


def _rate(outcomes: pl.DataFrame, need: str) -> float:
    df = outcomes.filter(pl.col("need") == need)
    assert df.height, f"no rows to measure a {need} rate on"
    return df["y"].mean()


# --------------------------------------------------------------------------- #
# truth.json is a contract too: the model can only recover what was planted.
# --------------------------------------------------------------------------- #

def test_truth_covers_every_term_in_the_design(truth: simulate.Truth) -> None:
    """A design term with no truth coefficient is a term the model is asked to recover from
    a world where it never acted. That is not a fit, it is a coin toss."""
    missing = sorted(set(design.TERM_BY_NAME) - set(truth.coefficients))
    assert not missing, f"truth.json has no coefficient for {missing}"
    unknown = sorted(set(truth.coefficients) - set(design.TERM_BY_NAME))
    assert not unknown, f"truth.json names terms design.py does not have: {unknown}"
    assert set(truth.sources) == set(truth.coefficients), (
        "every coefficient needs a one-line source; these are assumptions, not measurements, "
        "and a judge will ask where each one came from")


def test_truth_builds_a_coefficient_matrix_on_the_design_support(truth: simulate.Truth) -> None:
    B = truth.coef_matrix()
    assert B.shape == (design.P, design.K)
    assert np.isfinite(B).all()
    assert not B[~design.SUPPORT].any(), "a coefficient landed off the design's support"


def test_medication_interactions_are_planted(truth: simulate.Truth) -> None:
    """SPEC 5.4: plant them now or the model can never recover them."""
    for term in PLANTED_MED_TERMS:
        coef = truth.coefficients[term]
        values = list(coef.values()) if isinstance(coef, dict) else [coef]
        assert all(v > 0 for v in values), f"{term} is planted at zero, so it can never be found"


def test_latent_scales_are_positive(truth: simulate.Truth) -> None:
    assert truth.sigma_zip > 0 and truth.sigma_frailty > 0


# --------------------------------------------------------------------------- #
# The two acceptance tests from docs/SPEC.md 5.4
# --------------------------------------------------------------------------- #

def test_calm_day_rate_is_the_intercept(truth: simulate.Truth) -> None:
    """Hazards zeroed, person features zeroed: the daily rate per need is sigmoid(alpha_k).

    2,000 veterans x 60 days is ~180 events on the rarest need, so Monte Carlo noise is
    under 8% -- comfortably inside the 30% the acceptance test allows.
    """
    days = _days(60)
    cohort = _many(_vet(**NEUTRAL), 2_000, "N")
    hazards, sites = _world(cohort, days)

    out = simulate.simulate(cohort, hazards, sites, truth=truth, seed=0, latent=False)

    for need in NEEDS:
        want = _sigmoid(truth.coefficients["alpha"][need])
        got = _rate(out, need)
        assert 0.7 * want <= got <= 1.3 * want, (
            f"{need}: calm-day rate {got:.5f} is not within 30% of sigmoid(alpha) {want:.5f}")


def test_site_down_at_least_triples_the_treatment_gap_for_dialysis(truth: simulate.Truth) -> None:
    """The Sandy story as a number: station 630 closes and its dialysis patients are the
    ones who feel it. Dialysis patients of a site that stayed open must not move."""
    days = _days(60)
    closed = set(days[20:50])
    here = _many(_vet(**NEUTRAL | {"ckd_dialysis": True, "facility_id": "630"}), 1_000, "V630")
    away = _many(_vet(**NEUTRAL | {"ckd_dialysis": True, "facility_id": "630A4"}), 1_000, "VA4")
    cohort = pl.concat([here, away])
    hazards = _calm(cohort["modzcta"].unique().to_list(), days)
    sites = _sites(["630", "630A4"], days, down={"630": closed})

    out = simulate.simulate(cohort, hazards, sites, truth=truth, seed=0, latent=False)
    gap = out.filter(pl.col("need") == "treatment_gap").with_columns(
        down=pl.col("date").is_in(sorted(closed)),
        at_630=pl.col("veteran_id").str.starts_with("V630-"))

    up = gap.filter(pl.col("at_630") & ~pl.col("down"))["y"].mean()
    down = gap.filter(pl.col("at_630") & pl.col("down"))["y"].mean()
    assert down >= 3 * up, (
        f"the treatment-gap rate for dialysis patients at 630 went {up:.4f} -> {down:.4f} "
        "while their site was closed; a closure has to at least triple it")

    ctrl_up = gap.filter(~pl.col("at_630") & ~pl.col("down"))["y"].mean()
    ctrl_down = gap.filter(~pl.col("at_630") & pl.col("down"))["y"].mean()
    assert 0.7 <= ctrl_down / ctrl_up <= 1.4, (
        f"dialysis patients at 630A4, which stayed open, moved {ctrl_up:.4f} -> {ctrl_down:.4f}; "
        "a closure must reach only the site that closed")


# --------------------------------------------------------------------------- #
# The planted medication terms: dormant today, live the moment prompt 1 lands
# --------------------------------------------------------------------------- #

def test_medication_terms_contribute_nothing_while_the_columns_are_empty(
        truth: simulate.Truth) -> None:
    """With every medication column zero -- today's cohort -- zeroing those six coefficients
    cannot change a single outcome. That is what makes the medication and simulator prompts
    genuinely parallel, and it fails the moment a term leaks onto a non-medication feature."""
    days = _days(21)
    cohort = _many(_vet(**MED_COLUMNS), 300, "NOMEDS")
    hazards, sites = _world(cohort, days, heat_index_max_f=96.0, hot_day=True,
                            outage_frac=0.7, mail_delivery_disrupted=True)
    sites = sites.with_columns(site_down=pl.lit(True))

    zeroed = simulate.Truth(
        coefficients={**truth.coefficients, **dict.fromkeys(DORMANT_MED_TERMS, 0.0)},
        sources=truth.sources, sigma_zip=truth.sigma_zip,
        sigma_frailty=truth.sigma_frailty, name="zeroed")

    kw = dict(dates=days, seed=0)
    assert simulate.simulate(cohort, hazards, sites, truth=truth, **kw).equals(
        simulate.simulate(cohort, hazards, sites, truth=zeroed, **kw)), (
        "zeroing the medication interactions changed an outcome, so something other than a "
        "medication column is feeding them")


def test_medication_terms_switch_on_when_the_columns_arrive(truth: simulate.Truth) -> None:
    """The same veteran, the same heat wave, one prescription apart."""
    days = _days(30)
    plain = _many(_vet(**NEUTRAL), 1_500, "PLAIN")
    on_meds = _many(_vet(**NEUTRAL | {"med_thermoreg_score": 2.0}), 1_500, "MEDS")
    cohort = pl.concat([plain, on_meds])
    hazards, sites = _world(cohort, days, heat_index_max_f=96.0, hot_day=True, heat_alert=True)

    out = simulate.simulate(cohort, hazards, sites, truth=truth, seed=0, latent=False)
    heat = out.filter(pl.col("need") == "heat").with_columns(
        meds=pl.col("veteran_id").str.starts_with("MEDS-"))
    without = heat.filter(~pl.col("meds"))["y"].mean()
    with_meds = heat.filter(pl.col("meds"))["y"].mean()
    assert with_meds > 2 * without, (
        f"two heat-impairing medications moved the heat rate {without:.4f} -> {with_meds:.4f}; "
        "truth.json plants theta_heat_x_meds per score unit, so it should be several times")


# --------------------------------------------------------------------------- #
# The simulator on a whole cohort
# --------------------------------------------------------------------------- #

def test_off_event_rates_sit_in_the_spec_band(truth: simulate.Truth) -> None:
    """SPEC 5.5: outcome base rates per need are 0.2-2% per day off-event. A real cohort
    runs above sigmoid(alpha_k) because its betas are real; it must not run away."""
    days = _days(60)
    cohort = table("cohort")
    hazards, sites = _world(cohort, days)

    out = simulate.simulate(cohort, hazards, sites, truth=truth, seed=0)
    for need in NEEDS:
        got = _rate(out, need)
        assert 0.002 <= got <= 0.02, f"{need}: off-event rate {got:.4f} is outside 0.2-2%/day"


def test_a_hazard_raises_the_need_it_belongs_to_and_leaves_the_others_alone(
        truth: simulate.Truth) -> None:
    """Wildfire smoke is a breathing event, not a heat event. A design column joined to the
    wrong need would still look plausible in aggregate, so check the needs separately."""
    days = _days(30)
    cohort = _many(_vet(**NEUTRAL | {"copd": True, "pact_presumptive": True}), 1_500, "SMOKE")
    calm_h, sites = _world(cohort, days)
    smoky = calm_h.with_columns(pl.lit(203.5).alias("pm25"), pl.lit(True).alias("smoke_alert"))

    kw = dict(truth=truth, seed=0, latent=False)
    calm = simulate.simulate(cohort, calm_h, sites, **kw)
    smoke = simulate.simulate(cohort, smoky, sites, **kw)

    assert _rate(smoke, "breathing") > 3 * _rate(calm, "breathing"), (
        "the 7 June 2023 Queens peak barely moved breathing risk for a COPD veteran with "
        "PACT Act exposure")
    for need in ("heat", "mental", "access_loss"):
        moved = _rate(smoke, need) / _rate(calm, need)
        assert 0.7 <= moved <= 1.4, f"smoke moved {need} by {moved:.2f}x; it acts on breathing"


def test_draws_match_the_probabilities_they_were_drawn_from(truth: simulate.Truth) -> None:
    """Catches an off-by-one between the design rows and the uniforms drawn against them."""
    cohort, hazards, sites = table("cohort"), table("hazards"), table("site_status")
    dates = sorted(hazards["date"].unique().to_list())

    kw = dict(dates=dates, truth=truth, seed=0)
    out = simulate.simulate(cohort, hazards, sites, **kw)
    p = simulate.rates(cohort, hazards, sites, **kw)
    assert p.select("veteran_id", "date", "need").equals(out.select("veteran_id", "date", "need"))

    expected, realised = p["p"].sum(), out["y"].sum()
    assert abs(realised - expected) <= 0.08 * expected, (
        f"{realised} events drawn against {expected:.0f} expected")
    for need in NEEDS:
        want = p.filter(pl.col("need") == need)["p"].mean()
        assert abs(_rate(out, need) - want) <= 0.25 * want, (
            f"{need}: drew {_rate(out, need):.5f} against p {want:.5f}")


def test_quiet_dates_are_the_days_with_nothing_happening() -> None:
    days = _days(6)
    hazards = _calm(["10001", "10002"], days)
    hot = hazards.with_columns(
        hot_day=pl.col("date") == days[2],
        heat_index_max_f=pl.when(pl.col("date") == days[2]).then(96.0).otherwise(72.0))
    sites = _sites(["630"], days, down={"630": {days[4]}})
    assert simulate.quiet_dates(hot, sites) == [days[0], days[1], days[3], days[5]]


# --------------------------------------------------------------------------- #
# Determinism. The same click must produce the same number on stage.
# --------------------------------------------------------------------------- #

def test_the_same_seed_gives_the_same_outcomes(truth: simulate.Truth) -> None:
    cohort, hazards, sites = table("cohort").head(200), table("hazards"), table("site_status")
    dates = sorted(hazards["date"].unique().to_list())[:14]

    kw = dict(dates=dates, truth=truth)
    first = simulate.simulate(cohort, hazards, sites, seed=0, **kw)
    again = simulate.simulate(cohort, hazards, sites, seed=0, **kw)
    other = simulate.simulate(cohort, hazards, sites, seed=1, **kw)
    assert first.equals(again)
    assert first["y"].sum() != other["y"].sum(), "a different seed produced identical outcomes"


def test_a_day_draws_the_same_outcome_whatever_window_it_is_simulated_in(
        truth: simulate.Truth) -> None:
    """Outcomes are seeded per calendar day, so simulating a week does not re-roll the year
    -- and the answer cannot depend on how the run was chunked."""
    cohort, hazards, sites = table("cohort").head(200), table("hazards"), table("site_status")
    dates = sorted(hazards["date"].unique().to_list())[:21]

    whole = simulate.simulate(cohort, hazards, sites, dates=dates, truth=truth, seed=0)
    window = simulate.simulate(cohort, hazards, sites, dates=dates[7:14], truth=truth, seed=0)
    assert window.equals(whole.filter(pl.col("date").is_in(dates[7:14])))
    chunked = simulate.simulate(cohort, hazards, sites, dates=dates, truth=truth, seed=0,
                                days_per_build=5)
    assert chunked.equals(whole)


def test_the_latent_terms_are_the_same_people_every_run(truth: simulate.Truth) -> None:
    """A ZIP keeps its effect whether the whole cohort or one slice of it is simulated,
    because the model is being asked to recover the same world each time."""
    cohort = table("cohort")
    whole = simulate.latent_offset(cohort, truth, seed=0)
    part = simulate.latent_offset(cohort.head(50), truth, seed=0)
    assert np.allclose(whole[:50], part)
    assert not np.allclose(whole[:50], simulate.latent_offset(cohort.head(50), truth, seed=1))
    assert abs(whole.std() - np.hypot(truth.sigma_zip, truth.sigma_frailty)) < 0.1


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #

def test_outcomes_match_their_contract(truth: simulate.Truth) -> None:
    cohort, hazards, sites = table("cohort").head(120), table("hazards"), table("site_status")
    dates = sorted(hazards["date"].unique().to_list())[:9]

    out = simulate.simulate(cohort, hazards, sites, dates=dates, truth=truth, seed=0)
    schema.validate(out, "outcomes")
    assert out.height == cohort.height * len(dates) * len(NEEDS)
    assert set(out["y"].unique().to_list()) <= {0, 1}
    per = out.group_by("veteran_id", "date").agg(pl.col("need").n_unique().alias("k"))
    assert per["k"].min() == len(NEEDS)
    assert out["y"].sum() > 0, "120 veterans over 9 days with no outcome at all is not a draw"

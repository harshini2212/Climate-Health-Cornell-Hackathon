"""ablate.py -- what each block of the model is actually worth. Written before the module.

Two halves, and the second is the one that matters.

The first half is the claim the deck makes: drop the SiteDown block and the care team averts
strictly less harm on the days station 630 is closed. The fixture below makes that a
ranking question with a margin rather than an arithmetic accident -- ten veterans who are
only worth calling *because* their station is shut, and ten who outrank them the moment that
term is gone.

The second half is the one that catches a broken ablation. Zeroing a coefficient block must
change **nothing whatsoever** on the days that block was not active: not the harm averted,
not the scores, not a single driver chip. An ablation that re-draws the priors, re-seeds, or
zeroes the wrong slice of the design still moves those rows, and the first half would not
notice -- it would just report a slightly different number and look like a finding.

`test_an_empty_ablation_reproduces_the_shipped_scorer` pins the third thing: ablate.py scores
the same way `leeward/model/score_prior.py` does. If it drifts, the ablation table would be
measuring the difference between two scorers rather than the difference between two models.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import polars as pl
import pytest

from leeward import schema
from leeward.api.schemas import AblationRow
from leeward.eval import ablate
from leeward.eval import decision_quality as dq
from leeward.model import design, score_prior
from leeward.schema import NEEDS
from tables import table

# --------------------------------------------------------------------------- #
# A four-day world: station 630 is closed on two of them.
#
#   A (10)  in active cancer treatment at 630, no caregiver, young, nothing else wrong. On
#           a day 630 is open they sit below the Self-serve floor and are never called. On a
#           closed day psi_sitedown + theta_sitedown_x_sitedependent lift them past group C.
#   C (10)  patients of 526, which never closes: dialysis, active cancer treatment, no
#           caregiver, little money, diabetes, over 65. High risk every day, and the group
#           that takes the ten calls whenever A does not.
#
# Both groups are site-dependent and on no medication, so both are eligible for exactly the
# same actions. That matters more than it looks: `allocate` values a veteran's call on the
# risk left after their *higher*-valued actions, chosen or not, so a group with one extra
# high-tau action would rank below one without it at the same risk, and the ranking this
# fixture is built to flip would flip for the wrong reason.
#
# Only A has treatment gaps, and only on the closed days. C has breathing events every day,
# which are real (so a calm day averts a non-zero amount of harm and the "changes nothing"
# assertions are not 0 == 0) but worth less than A's: w*tau is 5 x 0.40 against 3 x 0.25.
#
# The budget is ten calls, so exactly one of the two groups is served on any given day.
# --------------------------------------------------------------------------- #

START = date(2026, 7, 1)
DAYS = [START + timedelta(days=i) for i in range(4)]
DOWN_DAYS = [DAYS[1], DAYS[2]]
CALM_DAYS = [DAYS[0], DAYS[3]]

DOWN_STATION = "630"
QUIET_STATION = "526"
ZIP = "10463"
K = 10
DRAWS = 200
SEED = 0

CALL_TAU = 0.40          # care_team_call on treatment_gap, tau.yaml
CALL_TAU_BREATHING = 0.25


def _neutral(col: schema.Column):
    """The blandest legal value for a column: no condition, no hazard, no exposure."""
    if col.values:
        return col.values[0]
    if col.dtype == pl.Boolean:
        return False
    if isinstance(col.dtype, pl.List):
        return []
    lo = col.bounds[0] if col.bounds else 0
    if col.dtype.is_integer():
        return int(lo)
    if col.dtype.is_float():
        return float(lo)
    return ""


def _cohort() -> pl.DataFrame:
    cols = schema.TABLES["cohort"].columns
    base = {c.name: _neutral(c) for c in cols} | {
        "name_display": "Ablation Fixture", "age": 40, "modzcta": ZIP, "borough": "Bronx",
        "floor": "upper", "caregiver": "informal_coresident", "income_band": "mid",
        "home_ac": True, "hvi": 1, "days_supply_remaining": 90,
    } | {f"{c.name}_synthetic": True for c in cols if c.synthetic}
    rows = []
    for i in range(10):
        rows.append(base | {                      # A: only the closed station makes them risky
            "veteran_id": f"A{i:02d}", "facility_id": DOWN_STATION,
            "active_cancer_tx": True, "caregiver": "none",
        })
        rows.append(base | {                      # C: risky every day, never at a closed site
            "veteran_id": f"C{i:02d}", "facility_id": QUIET_STATION,
            "ckd_dialysis": True, "active_cancer_tx": True, "diabetes": True, "age": 70,
            "caregiver": "none", "low_assets": True, "n_chronic": 4,
        })
    return pl.DataFrame(rows, schema_overrides={c.name: c.dtype for c in cols})


def _hazards() -> pl.DataFrame:
    """One ZIP, four identical calm days: the only thing that varies is whether 630 is open."""
    cols = schema.TABLES["hazards"].columns
    base = {c.name: _neutral(c) for c in cols} | {"heat_index_max_f": 70.0, "pm25": 5.0}
    return pl.DataFrame([base | {"modzcta": ZIP, "date": d} for d in DAYS],
                        schema_overrides={c.name: c.dtype for c in cols})


def _site_status() -> pl.DataFrame:
    cols = schema.TABLES["site_status"].columns
    rows = [{"facility_id": f, "date": d, "site_down": f == DOWN_STATION and d in DOWN_DAYS,
             "evac_zone": 0, "site_dependent_services": True}
            for f in (DOWN_STATION, QUIET_STATION) for d in DAYS]
    return pl.DataFrame(rows, schema_overrides={c.name: c.dtype for c in cols})


def _outcomes() -> pl.DataFrame:
    """A's treatment gaps land only on the closed days; C's breathing events land every day."""
    rows = []
    for v in _cohort()["veteran_id"]:
        for d in DAYS:
            for need in NEEDS:
                y = ((v.startswith("A") and need == "treatment_gap" and d in DOWN_DAYS)
                     or (v.startswith("C") and need == "breathing"))
                rows.append({"veteran_id": v, "date": d, "need": need, "y": int(y)})
    return pl.DataFrame(rows, schema={"veteran_id": pl.Utf8, "date": pl.Date,
                                      "need": pl.Utf8, "y": pl.Int32})


def _world() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    cohort, hazards, sites, outcomes = _cohort(), _hazards(), _site_status(), _outcomes()
    for df, name in ((cohort, "cohort"), (hazards, "hazards"),
                     (sites, "site_status"), (outcomes, "outcomes")):
        schema.validate(df, name)
    return cohort, hazards, sites, outcomes


#: The SiteDown ablations, each of which must move the closed days and only the closed days.
SITEDOWN_KEYS = ("psi_sitedown", "theta_sitedown_x_sitedependent", "sitedown")


def _result(keys=SITEDOWN_KEYS) -> ablate.Result:
    cohort, hazards, sites, outcomes = _world()
    chosen = tuple(a for a in ablate.ABLATIONS if a.key in keys)
    return ablate.run(cohort, hazards, sites, outcomes, dates=DAYS, ablations=chosen,
                      k=K, n_draws=DRAWS, seed=SEED)


def _harm(daily: pl.DataFrame, dropped: str, day: date) -> float:
    row = daily.filter((pl.col("dropped") == dropped) & (pl.col("date") == day))
    assert row.height == 1, f"{dropped} on {day}: expected one row, got {row.height}"
    return float(row["harm_averted"][0])


# --------------------------------------------------------------------------- #
# The named set of terms
# --------------------------------------------------------------------------- #

def test_every_ablation_names_features_that_exist_in_the_design() -> None:
    known = set(design.FEATURES)
    assert ablate.ABLATIONS, "no ablations are defined"
    for a in ablate.ABLATIONS:
        assert a.features, f"{a.key} drops nothing"
        unknown = sorted(set(a.features) - known)
        assert not unknown, f"{a.key} names features the design does not have: {unknown}"


def test_the_four_blocks_the_deck_makes_claims_about_are_all_ablated() -> None:
    """SiteDown on its own, the interaction on its own, the medication block, the heat lag."""
    keys = {a.key for a in ablate.ABLATIONS}
    assert {"psi_sitedown", "theta_sitedown_x_sitedependent",
            "medications", "heat_lags"} <= keys


def test_labels_are_unique_because_dropped_is_the_tables_key() -> None:
    labels = [ablate.FULL] + [a.label for a in ablate.ABLATIONS]
    assert len(set(labels)) == len(labels), f"duplicate `dropped` labels: {labels}"


def test_the_medication_block_is_every_term_that_reads_a_medication_column() -> None:
    """Derived from the design, not hand-listed, so a term landing in C3/C4 joins it."""
    block = dict(ablate.MEDICATION_TERMS)
    for name in ("theta_heat_x_meds", "theta_heat_x_raas_diuretic", "theta_heat_x_acb",
                 "theta_heat_x_renal_triple", "theta_outage_x_cold_chain",
                 "theta_sitedown_x_controlled", "theta_mail_x_supply"):
        assert name in block, f"the medication block is missing {name}"
    # ...and nothing that only looks medical: no-AC and powered equipment are not drugs.
    for name in ("theta_heat_x_no_ac", "theta_outage_x_equipment", "psi_sitedown"):
        assert name not in block, f"{name} is not a medication term"


def test_the_climate_ablation_drops_every_term_that_is_zero_on_a_calm_day() -> None:
    """SPEC §11's first row: what the model is worth if it has never heard of weather."""
    climate = next(a for a in ablate.ABLATIONS if a.key == "climate")
    for name in ("delta_heat", "eps_pm25", "psi_sitedown", "zeta_flood", "kappa_outage",
                 "theta_sitedown_x_sitedependent", "theta_mail_x_supply"):
        assert set(design.TERM_BY_NAME[name].feature_names) <= set(climate.features), name
    # ...and nothing a veteran carries every day of the year.
    for name in ("alpha", "beta_copd", "beta_dialysis", "sigma_no_caregiver"):
        assert not set(design.TERM_BY_NAME[name].feature_names) & set(climate.features), name


def test_the_heat_lag_ablation_keeps_the_same_day_term() -> None:
    """Dropping delta_heat entirely would measure "does heat matter", a different question."""
    heat = next(a for a in ablate.ABLATIONS if a.key == "heat_lags")
    assert "delta_heat[0]" not in heat.features
    assert set(heat.features) == {f"delta_heat[{i}]" for i in range(1, design.HEAT_LAGS)}


# --------------------------------------------------------------------------- #
# Scoring the same way the shipped scorer does
# --------------------------------------------------------------------------- #

def test_an_empty_ablation_reproduces_the_shipped_scorer() -> None:
    """Otherwise the table measures two scorers against each other, not two models."""
    cohort, hazards, sites = table("cohort"), table("hazards"), table("site_status")
    days = sorted(hazards["date"].unique().to_list())[:2]
    mine = ablate.score_ablated(cohort, hazards, sites, drop=(), dates=days,
                                n_draws=40, seed=SEED)
    theirs = score_prior.score(cohort, hazards, sites, dates=days, n_draws=40, seed=SEED)
    assert mine.equals(theirs), "ablate.py and score_prior.py disagree about the full model"


def test_a_dropped_block_is_gone_from_the_linear_predictor() -> None:
    """The mechanical claim: zeroing the block is zeroing those rows of B, nothing else."""
    cohort, hazards, sites, _ = _world()
    full = ablate.score_ablated(cohort, hazards, sites, drop=(), dates=DOWN_DAYS,
                                n_draws=DRAWS, seed=SEED)
    dropped = ablate.score_ablated(cohort, hazards, sites, drop=("psi_sitedown",),
                                   dates=DOWN_DAYS, n_draws=DRAWS, seed=SEED)
    key = ["veteran_id", "date", "need"]
    both = full.join(dropped.select(*key, p_dropped="p_mean"), on=key)
    a_gap = both.filter(pl.col("veteran_id").str.starts_with("A")
                        & (pl.col("need") == "treatment_gap"))
    assert (a_gap["p_dropped"] < a_gap["p_mean"]).all(), (
        "dropping psi_sitedown did not lower risk for patients of the closed station")
    c_rows = both.filter(pl.col("veteran_id").str.starts_with("C"))
    assert (c_rows["p_dropped"] == c_rows["p_mean"]).all(), (
        "dropping psi_sitedown moved veterans whose station never closed")


# --------------------------------------------------------------------------- #
# The fixture's own ordering. Guards the two tests below: if the priors move and
# the three groups stop straddling each other, this fails first and says so.
# --------------------------------------------------------------------------- #

def test_the_fixture_puts_c_between_the_full_and_the_ablated_sitedown_groups() -> None:
    cohort, hazards, sites, _ = _world()

    def gap(drop: tuple[str, ...], prefix: str) -> float:
        s = ablate.score_ablated(cohort, hazards, sites, drop=drop, dates=[DOWN_DAYS[0]],
                                 n_draws=DRAWS, seed=SEED)
        return float(s.filter(pl.col("veteran_id").str.starts_with(prefix)
                              & (pl.col("need") == "treatment_gap"))["p_mean"].mean())

    c = gap((), "C")
    assert gap((), "A") > c, "group A does not outrank C on a closed day with the full model"
    for a in ablate.ABLATIONS:
        if a.key in SITEDOWN_KEYS:
            assert gap(a.features, "A") < c, (
                f"dropping {a.key} left group A above C, so the call list would not change")


# --------------------------------------------------------------------------- #
# Half one: the closed days
# --------------------------------------------------------------------------- #

def test_dropping_sitedown_lowers_harm_averted_on_the_days_630_is_down() -> None:
    result = _result()
    for day in DOWN_DAYS:
        full = _harm(result.daily, ablate.FULL, day)
        assert full == pytest.approx(K * 5 * CALL_TAU), (
            f"the full model should be calling all ten patients of 630 on {day}")
        for a in ablate.ABLATIONS:
            if a.key not in SITEDOWN_KEYS:
                continue
            assert _harm(result.daily, a.label, day) < full, (
                f"dropping {a.key} averted as much harm on {day} as keeping it")


def test_the_sitedown_rows_of_the_table_cost_harm_averted_overall() -> None:
    result = _result()
    full = float(result.table.filter(pl.col("dropped") == ablate.FULL)["harm_averted_at_k"][0])
    for a in ablate.ABLATIONS:
        if a.key not in SITEDOWN_KEYS:
            continue
        row = result.table.filter(pl.col("dropped") == a.label)
        assert row.height == 1
        assert float(row["harm_averted_at_k"][0]) < full
        assert float(row["d_harm_averted"][0]) < 0, (
            f"{a.key}'s delta should be negative: the block is worth something")


# --------------------------------------------------------------------------- #
# Half two: the days no site is down. The one that catches a broken ablation.
# --------------------------------------------------------------------------- #

def test_dropping_sitedown_changes_nothing_on_days_no_site_is_down() -> None:
    """X is zero in that block on a calm day, so the ablated model *is* the full model.

    A re-drawn or re-seeded prior, or the wrong slice of the design, moves these rows too.
    """
    result = _result()
    for day in CALM_DAYS:
        full = _harm(result.daily, ablate.FULL, day)
        assert full == pytest.approx(K * 3 * CALL_TAU_BREATHING), (
            f"the calm day {day} must avert a non-zero amount of harm, or this test is 0 == 0")
        for a in ablate.ABLATIONS:
            if a.key in SITEDOWN_KEYS:
                assert _harm(result.daily, a.label, day) == full, (
                    f"dropping {a.key} moved {day}, when no station was closed")


def test_a_calm_days_scores_are_identical_down_to_the_driver_chips() -> None:
    """Sharper than harm averted: nothing in the row may move, not even a tie-broken driver."""
    cohort, hazards, sites, _ = _world()
    full = ablate.score_ablated(cohort, hazards, sites, drop=(), dates=DAYS,
                                n_draws=DRAWS, seed=SEED)
    for a in ablate.ABLATIONS:
        if a.key not in SITEDOWN_KEYS:
            continue
        got = ablate.score_ablated(cohort, hazards, sites, drop=a.features, dates=DAYS,
                                   n_draws=DRAWS, seed=SEED)
        calm = pl.col("date").is_in(CALM_DAYS)
        assert got.filter(calm).equals(full.filter(calm)), (
            f"dropping {a.key} changed a calm day's scores")
        assert not got.filter(~calm).equals(full.filter(~calm)), (
            f"dropping {a.key} changed nothing on the closed days either")


# --------------------------------------------------------------------------- #
# What comes out
# --------------------------------------------------------------------------- #

def test_the_table_leads_with_the_full_model_and_fits_the_report_contract() -> None:
    result = _result()
    assert result.table["dropped"][0] == ablate.FULL
    assert result.table.height == 1 + len(SITEDOWN_KEYS)
    full = result.table.row(0, named=True)
    assert full["d_ece"] == 0.0 and full["d_harm_averted"] == 0.0
    for row in ablate.rows(result.table):
        AblationRow(**row)


def test_the_ece_column_is_the_pooled_holdout_number() -> None:
    from leeward.eval import calibration as cal
    cohort, hazards, sites, outcomes = _world()
    result = _result()
    scores = ablate.score_ablated(cohort, hazards, sites, drop=(), dates=DAYS,
                                  n_draws=DRAWS, seed=SEED)
    expected = cal.ece_overall(cal.run(scores, outcomes, dates=DAYS))
    got = float(result.table.filter(pl.col("dropped") == ablate.FULL)["ece"][0])
    assert got == pytest.approx(expected)


def test_harm_averted_at_40_matches_the_impact_charts_own_allocator() -> None:
    """Same allocator, same w and tau as decision_quality: one number, not two."""
    cohort, hazards, sites, outcomes = _world()
    result = _result()
    scores = ablate.score_ablated(cohort, hazards, sites, drop=(), dates=DAYS,
                                  n_draws=DRAWS, seed=SEED)
    w, tau = dq.decision_weights()
    tidy = dq.evaluate(scores, cohort, outcomes, w=w, tau=tau, ks=(K,), dates=DAYS)
    expected = float(tidy.filter(pl.col("strategy") == "leeward")["harm_averted"].mean())
    got = float(result.table.filter(pl.col("dropped") == ablate.FULL)["harm_averted_at_k"][0])
    assert got == pytest.approx(expected)


def test_outputs_are_a_tidy_csv_a_json_payload_and_an_offline_chart(tmp_path) -> None:
    result = _result()
    written = ablate.write_outputs(result, tmp_path)
    assert [p.name for p in written] == ["ablations.csv", "ablations.json", "ablations.html"]
    back = pl.read_csv(written[0])
    assert back.columns == list(ablate.TABLE_SCHEMA)
    assert back.height == result.table.height
    page = written[2].read_text(encoding="utf-8")
    assert "plotly" in page.lower()
    assert not re.search(r"<script[^>]*\bsrc=[\"']https?://", page), (
        "the chart loads plotly.js from the network; it must open with the wifi off")


def test_an_unknown_feature_is_refused_rather_than_silently_ablating_nothing() -> None:
    cohort, hazards, sites, _ = _world()
    with pytest.raises(ValueError, match="not a design feature"):
        ablate.score_ablated(cohort, hazards, sites, drop=("psi_sitedown_typo",),
                             dates=DAYS, n_draws=DRAWS, seed=SEED)

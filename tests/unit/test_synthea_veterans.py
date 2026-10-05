"""Acceptance tests for `leeward/cohort/synthea.py` -- Synthea's veteran modules in the cohort.

Written before the module. Track A needs PTSD, depression, suicide risk, substance use, cancer
and homelessness to come from one generated patient, not from separate coin flips, so that
a veteran's diagnoses and their prescriptions belong to the same person. These tests hold
the cohort to the distilled Synthea table (`synthea_veteran_profiles.parquet`), never to
numbers build.py computed for itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from leeward import schema
from leeward.cohort import build, medications, synthea

REF = schema.REFERENCE
N = 10_000

#: Track A's condition columns. Each is one Synthea module (or module family).
FIELDS = ("ptsd", "depression", "active_cancer_tx", "suicide_risk",
          "substance_use_disorder", "homeless")
NEW_COLUMNS = ("suicide_risk", "substance_use_disorder", "homeless")

#: Flags the draw is tilted to a cited rate (build.py, "Track A"). The rest take Synthea's own.
WEIGHTED = ("ptsd", "depression", "substance_use_disorder")
UNWEIGHTED = tuple(f for f in FIELDS if f not in WEIGHTED)

#: The stated tolerance: the cohort's rate must sit within this many binomial standard
#: errors of its source -- the cited rate for a weighted flag, Synthea's own rate
#: (reweighted to the cohort's sex x age-band mix) for the others.
SE_TOLERANCE = 3.0


@pytest.fixture(scope="module")
def profiles() -> pl.DataFrame:
    return synthea.load_profiles()


@pytest.fixture(scope="module")
def cohort() -> pl.DataFrame:
    return build.build(n=N, seed=0)


def _stratum(age: pl.Expr) -> pl.Expr:
    return synthea.band_expr(age).alias("band")


# --------------------------------------------------------------------------- #
# The committed profile table
# --------------------------------------------------------------------------- #

def test_profiles_are_one_row_per_living_adult_synthea_patient(profiles: pl.DataFrame) -> None:
    assert profiles["profile_id"].n_unique() == profiles.height
    assert profiles.height >= 7_000, f"only {profiles.height} profiles; the run looks truncated"
    assert profiles["age"].min() >= 18 and profiles["age"].max() <= 110
    assert set(profiles["sex"].unique()) == {"M", "F"}
    for f in FIELDS:
        assert profiles[f].dtype == pl.Boolean, f"{f} must be a flag"
    assert profiles["rxcuis"].dtype == pl.List(pl.Utf8)


def test_every_track_a_condition_actually_occurs_in_the_source(profiles: pl.DataFrame) -> None:
    """A module that never fires would leave a column False for everyone and pass every
    other test here. Each one must be present in the generated population."""
    for f in FIELDS:
        assert profiles[f].sum() >= 10, f"{f}: {profiles[f].sum()} profiles -- did the module run?"


def test_every_sex_and_age_band_has_enough_profiles_to_bootstrap(profiles: pl.DataFrame) -> None:
    cells = profiles.with_columns(_stratum(pl.col("age"))).group_by("sex", "band").len()
    assert cells.height == 2 * len(build.BANDS), f"missing strata: {cells}"
    assert cells["len"].min() >= 50, f"a stratum is too thin to bootstrap from:\n{cells}"


def test_the_run_is_pinned_and_cited() -> None:
    """Version, jar hash, seed and the veteran override, all recorded where a reader looks."""
    assert synthea.SYNTHEA_VERSION == "v4.0.0"
    assert len(synthea.JAR_SHA256) == 64
    args = " ".join(synthea.RUN_ARGS)
    for flag in ("-s 0", "-cs 0", "-r 20260101", "generate.veteran_population_override=true"):
        assert flag in args, f"the Synthea run must pin {flag}"
    sources = (Path(__file__).resolve().parents[2] / "docs" / "sources.md").read_text()
    assert "synthea_veteran_profiles" in sources and synthea.SYNTHEA_VERSION in sources
    assert synthea.JAR_SHA256 in sources


def test_the_condition_codes_are_the_ones_the_modules_emit() -> None:
    """Spot-check the SNOMED codes against the Synthea v4.0.0 module files they came from."""
    assert "47505003" in synthea.CONDITION_CODES["ptsd"]           # veteran_ptsd.json
    assert "370143000" in synthea.CONDITION_CODES["depression"]    # veteran_mdd.json
    assert "32911000" in synthea.CONDITION_CODES["homeless"]       # homelessness.json
    assert "7200002" in synthea.CONDITION_CODES["substance_use_disorder"]
    assert "225444004" in synthea.CONDITION_CODES["suicide_risk"]  # veteran_ptsd.json
    # Completed suicide is an event on a dead patient; it is not a risk to act on.
    assert "44301001" not in synthea.CONDITION_CODES["suicide_risk"]


# --------------------------------------------------------------------------- #
# The CSV distiller, on a hand-built record
# --------------------------------------------------------------------------- #

def _write_csvs(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "Id": ["alive", "dead", "child"],
        "BIRTHDATE": ["1950-06-01", "1950-06-01", "2015-01-01"],
        "DEATHDATE": [None, "2020-01-01", None],
        "GENDER": ["M", "F", "F"],
    }).write_csv(tmp / "patients.csv")
    pl.DataFrame({
        "START": ["2018-01-01", "2010-01-01", "2019-01-01", "2001-01-01", "2005-01-01",
                  "2016-01-01", "2016-01-01"],
        "STOP": ["2020-01-01", "2011-01-01", "2019-02-01", None, None, None, None],
        "PATIENT": ["alive", "alive", "alive", "dead", "child", "alive", "alive"],
        "CODE": ["47505003", "370143000", "225444004", "47505003", "32911000",
                 "7200002", "126906006"],
        "DESCRIPTION": ["Posttraumatic stress disorder (disorder)",
                        "Major depressive disorder (disorder)",
                        "At increased risk for suicide (finding)",
                        "Posttraumatic stress disorder (disorder)", "Homeless (finding)",
                        "Alcoholism (disorder)", "Neoplasm of prostate (disorder)"],
    }).write_csv(tmp / "conditions.csv")
    pl.DataFrame({   # radiotherapy 2 years ago: the prostate cancer is no longer in treatment
        "START": ["2024-01-01"], "PATIENT": ["alive"], "CODE": ["999"],
    }).write_csv(tmp / "procedures.csv")
    pl.DataFrame({
        "START": ["2020-01-01", "2010-01-01"],
        "STOP": [None, "2011-01-01"],
        "PATIENT": ["alive", "alive"],
        "CODE": ["312938", "310385"],
    }).write_csv(tmp / "medications.csv")
    return tmp


def test_the_distiller_keeps_active_conditions_and_meds_and_drops_the_dead(tmp_path: Path) -> None:
    p = synthea.profiles_from_csv(_write_csvs(tmp_path / "csv"),
                                  antineoplastic_rxcuis=frozenset(), cancer_tx_procedures={"999"})
    assert p["profile_id"].to_list() == ["alive"], "dead and under-18 patients are not veterans to reach"
    row = p.row(0, named=True)
    assert row["age"] == 75 and row["sex"] == "M"
    assert row["ptsd"] is True, "PTSD in the record stays on the problem list after treatment"
    assert row["depression"] is False, "a depression episode that ended in 2011 is not active"
    assert row["suicide_risk"] is True, "a suicide-risk finding in the record counts"
    assert row["substance_use_disorder"] is True
    assert row["active_cancer_tx"] is False, "an open cancer with no treatment in 365 days"
    assert row["rxcuis"] == ["312938"], "only medications still active at the reference date"


# --------------------------------------------------------------------------- #
# The cohort, held to the source
# --------------------------------------------------------------------------- #

def test_new_columns_are_in_the_contract_and_pass_the_writer(cohort: pl.DataFrame) -> None:
    contract = {c.name: c for c in schema.TABLES["cohort"].columns}
    for col in NEW_COLUMNS:
        assert col in contract, f"{col} is not in leeward/schema.py"
        assert cohort[col].dtype == pl.Boolean
    schema.validate(cohort, "cohort")


def test_schema_write_accepts_the_new_cohort(cohort: pl.DataFrame, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schema, "DATA", tmp_path)   # Table.path reads schema.DATA
    out = schema.write(cohort, "cohort")
    assert out == tmp_path / "cohort.parquet"
    assert set(NEW_COLUMNS) <= set(pl.read_parquet(out).columns)


@pytest.mark.parametrize("field", UNWEIGHTED)
def test_unweighted_prevalence_matches_the_synthea_source_within_three_standard_errors(
        cohort: pl.DataFrame, profiles: pl.DataFrame, field: str) -> None:
    """Uniform draw within sex x age band is exact in expectation: the source rate in each
    stratum, weighted by how many veterans the cohort put there. Tolerance: 3 binomial SE."""
    src = (profiles.with_columns(_stratum(pl.col("age")))
                   .group_by("sex", "band").agg(pl.col(field).mean().alias("rate")))
    mix = (cohort.with_columns(_stratum(pl.col("age")))
                 .group_by("sex", "band").len()
                 .join(src, on=["sex", "band"], how="left"))
    assert mix["rate"].null_count() == 0, "a cohort stratum has no Synthea profiles"
    expected = float((mix["len"] * mix["rate"]).sum() / mix["len"].sum())
    _within_se(field, float(cohort[field].mean()), expected, cohort.height, "Synthea source")


def _cited_target(cohort: pl.DataFrame, field: str) -> np.ndarray:
    """Each veteran's cited rate, read straight from the sources -- not from build.py's frame."""
    if field == "depression":
        places = pl.read_parquet(REF / "places_zcta_nyc.parquet").select(
            pl.col("zcta").alias("geo_id"), (pl.col("depression_crudeprev") / 100).alias("t"))
        return cohort.select("geo_id").join(places, on="geo_id", how="left")["t"].to_numpy()
    table = {"ptsd": build.PTSD_PAST_YEAR, "substance_use_disorder": build.SUD_PAST_YEAR}[field]
    return np.vectorize(table.get)(cohort["deployment_era"].to_numpy())


@pytest.mark.parametrize("field", WEIGHTED)
def test_weighted_prevalence_matches_the_cited_rate_within_three_standard_errors(
        cohort: pl.DataFrame, field: str) -> None:
    """PTSD to VA NCPTSD by era, depression to PLACES per ZIP, substance use to NSDUH."""
    target = _cited_target(cohort, field)
    assert not np.isnan(target).any(), f"veterans with no cited {field} rate"
    _within_se(field, float(cohort[field].mean()), float(target.mean()), cohort.height,
               "cited rate")


def test_depression_keeps_the_places_zip_gradient(cohort: pl.DataFrame) -> None:
    """The cost 6b asked us to measure: a flat Synthea draw loses PLACES' per-ZIP gradient.
    The tilt restores it. Before Track A, per-ZIP r was 0.42 (ZIPs with >= 30 veterans)."""
    by = cohort.group_by("geo_id").agg(pl.len().alias("n"), pl.col("depression").mean())
    places = pl.read_parquet(REF / "places_zcta_nyc.parquet").select(
        pl.col("zcta").alias("geo_id"), "depression_crudeprev")
    j = by.filter(pl.col("n") >= 30).join(places, on="geo_id")
    r = np.corrcoef(j["depression"].to_numpy(), j["depression_crudeprev"].to_numpy())[0, 1]
    assert r > 0.3, f"per-ZIP depression correlates {r:.2f} with PLACES"


def _within_se(field: str, realised: float, expected: float, n: int, what: str) -> None:
    se = np.sqrt(max(expected * (1 - expected), 1e-6) / n)
    assert abs(realised - expected) <= SE_TOLERANCE * se, (
        f"{field}: cohort {realised:.4f}, {what} {expected:.4f}, "
        f"|diff| {abs(realised - expected):.4f} > {SE_TOLERANCE} SE ({SE_TOLERANCE * se:.4f})")


def test_a_thin_pool_borrows_only_carriers_and_only_from_the_nearest_band() -> None:
    """Synthea gives men now 65-74 almost no PTSD: it makes them peacetime veterans. The tilt
    cannot reach a rate no profile carries, so a pool short of carriers (< MIN_CARRIERS)
    borrows *carriers of that flag* from the nearest band of the same sex. Everyone else
    still gets a profile from their own band, and every profile stays whole."""
    n = 4000
    sex = np.array(["M"] * n)
    age = np.full(n, 70)
    got = synthea.draw(sex, age, seed=0, targets={"ptsd": np.full(n, 0.05)})
    own_band = got.with_columns(_stratum(pl.col("age")))["band"] == "65_74"
    assert got["ptsd"].mean() > 0.03, f"PTSD {got['ptsd'].mean():.3f}; the pool was never widened"
    assert got.filter(~own_band)["ptsd"].all(), "a non-carrier was borrowed from another band"
    borrowed = set(got.filter(~own_band).with_columns(_stratum(pl.col("age")))["band"])
    assert borrowed <= {"55_64", "75plus"}, f"borrowed from {borrowed}, not the nearest bands"
    assert (~own_band).mean() < 0.1


def test_diagnoses_and_medications_come_from_the_same_patient(
        cohort: pl.DataFrame, profiles: pl.DataFrame) -> None:
    """The point of the swap. Every veteran's condition flags *and* medication list are one
    Synthea patient's, of the same sex -- never stitched from two people. (The band can be
    a neighbour's for a borrowed carrier; see the test above.)"""
    key = lambda df: {  # noqa: E731
        (r["sex"], *(r[f] for f in FIELDS), tuple(r["rxcuis"]))
        for r in df.iter_rows(named=True)}
    real = key(profiles)
    got = key(cohort.rename({"med_rxcuis": "rxcuis"}))
    stitched = got - real
    assert not stitched, f"{len(stitched)} veterans carry a combination no Synthea patient has"


def test_a_psychiatric_diagnosis_brings_its_medication_with_it(cohort: pl.DataFrame) -> None:
    """The known gap the parametric draw left open: diagnosis and prescription were
    independent. veteran_ptsd / veteran_mdd prescribe SSRIs, so the antidepressant classes
    must now be far more common among veterans with PTSD or depression than without."""
    on_ad = cohort["va_drug_classes"].list.eval(
        pl.element().is_in(["CN600", "CN609", "CN601", "CN602"])).list.any()
    dx = cohort["ptsd"] | cohort["depression"]
    with_dx, without = on_ad.filter(dx).mean(), on_ad.filter(~dx).mean()
    assert with_dx is not None and without is not None
    assert with_dx > 2 * without, (
        f"antidepressant use {with_dx:.1%} with PTSD/depression vs {without:.1%} without")


def test_a_homeless_veteran_has_no_home_to_cool(cohort: pl.DataFrame) -> None:
    """Approved 2026-10-05: home_ac False, floor ground, lives_alone True."""
    h = cohort.filter(pl.col("homeless"))
    assert h.height > 0
    assert not h["home_ac"].any()
    assert (h["floor"] == "ground").all()
    assert h["lives_alone"].all()


def test_medication_lists_come_from_the_veteran_profiles(cohort: pl.DataFrame,
                                                        profiles: pl.DataFrame) -> None:
    real = {tuple(r) for r in profiles["rxcuis"].to_list()}
    assert {tuple(r) for r in cohort["med_rxcuis"].to_list()} <= real
    # And the derived block still agrees with the codes.
    again = medications.derive(cohort["med_rxcuis"].to_list())
    assert cohort["med_thermoreg_score"].to_list() == again["med_thermoreg_score"].to_list()


def test_same_seed_same_patients_different_seed_different_ones() -> None:
    a, b, c = (build.build(n=400, seed=s) for s in (0, 0, 1))
    cols = list(FIELDS) + ["med_rxcuis"]
    assert a.select(cols).equals(b.select(cols))
    assert not a.select(cols).equals(c.select(cols))

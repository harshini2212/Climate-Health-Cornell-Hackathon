"""The vendored public data must be present, intact, and joinable.

This is the one test suite that passes on a clean clone before any of the build exists.
If it goes red, something changed `data/reference/` and every downstream prior is suspect.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
MANIFEST = json.loads((REF / "manifest.json").read_text())

# The five tables the cohort joins on modzcta / zcta. Anything here going missing
# means an augment prior silently falls back to an invented number.
ZIP_LEVEL = {
    "nyc_modzcta.parquet": "modzcta",
    "hvi_by_zcta.parquet": "zcta",
    "places_zcta_nyc.parquet": "zcta",
    "evac_zone_by_modzcta.parquet": "modzcta",
    "stormwater_by_modzcta.parquet": "modzcta",
}


@pytest.mark.parametrize("name", sorted(MANIFEST))
def test_manifest_entry_is_intact(name: str) -> None:
    entry = MANIFEST[name]
    path = REF / entry["file"]
    if entry["file"].endswith(".zip"):  # raw download, not committed
        pytest.skip(f"{entry['file']} lives in data/raw and is not committed")
    assert path.exists(), f"{path} is in the manifest but missing from the repo"
    assert path.stat().st_size == entry["bytes"], f"{path.name} size changed since fetch"
    if entry.get("sha256"):
        got = hashlib.sha256(path.read_bytes()).hexdigest()[: len(entry["sha256"])]
        assert got == entry["sha256"], f"{path.name} content changed since fetch"
    df = pl.read_parquet(path)
    assert df.height == entry["rows"], f"{path.name}: {df.height} rows, manifest says {entry['rows']}"


def test_modzcta_is_the_join_key() -> None:
    """Every ZIP-level table must cover NYC's 178 MODZCTAs."""
    base = set(pl.read_parquet(REF / "nyc_modzcta.parquet")["modzcta"].to_list())
    assert len(base) == 178

    for file, key in ZIP_LEVEL.items():
        keys = set(pl.read_parquet(REF / file)[key].cast(pl.Utf8).to_list())
        missing = base - keys
        # HVI and PLACES are published per ZCTA5; a couple of MODZCTAs merge ZCTAs that
        # those sources suppress for small population. More than five is a broken join.
        assert len(missing) <= 5, f"{file} misses {len(missing)} MODZCTAs: {sorted(missing)[:10]}"


def test_places_gradient_is_monotone_in_hvi() -> None:
    """CDC PLACES has never heard of the Heat Vulnerability Index.

    If these gradients stop being monotone, the join broke -- they are computed from
    independent sources and agreeing is the whole point.
    """
    mz = pl.read_parquet(REF / "nyc_modzcta.parquet").select("modzcta")
    hvi = pl.read_parquet(REF / "hvi_by_zcta.parquet")
    places = pl.read_parquet(REF / "places_zcta_nyc.parquet")

    j = (mz.join(hvi, left_on="modzcta", right_on="zcta", how="inner")
           .join(places, left_on="modzcta", right_on="zcta", how="inner"))

    for col in ["shututility_crudeprev", "mobility_crudeprev",
                "lacktrpt_crudeprev", "emotionspt_crudeprev"]:
        by_band = (j.group_by("hvi").agg(pl.col(col).mean().alias("m"))
                    .sort("hvi")["m"].to_list())
        assert by_band == sorted(by_band), f"{col} is not increasing across HVI bands: {by_band}"


def test_manhattan_va_is_in_evacuation_zone_one() -> None:
    """The SiteDown term's entire justification. If this changes, the pitch changes."""
    fac = pl.read_parquet(REF / "va_facilities_nyc_hazard.parquet")
    assert fac.height == 14
    row = fac.filter(pl.col("station_no") == "630")
    assert row.height == 1, "VA station 630 (Manhattan) missing from the facility table"
    assert row["evac_zone"][0] == 1, "station 630 is no longer in evacuation zone 1"
    assert row["site_dependent_services"][0] is True


def test_smoke_replay_has_the_june_2023_peak() -> None:
    """The smoke scenario replays real monitor data, so the peak must actually be there."""
    df = pl.read_parquet(REF / "airnow_pm25_nyc_smoke2023.parquet")
    peak = df.filter(pl.col("date") == "06/07/23")["value"].max()
    assert peak > 190, f"7 June 2023 peak PM2.5 is {peak}, expected ~203.5"
    baseline = df.filter(pl.col("date") == "06/05/23")["value"].max()
    assert baseline < 25, f"5 June 2023 baseline is {baseline}, expected ~13"


def test_acs_age_bands_sum_to_the_veteran_total() -> None:
    """Reading the wrong B21001 cells is silent and ruins the re-homing weights."""
    df = pl.read_parquet(REF / "acs_veterans_by_zcta.parquet")
    bands = ["vet_18_34", "vet_35_54", "vet_55_64", "vet_65_74", "vet_75plus"]
    total = df["veterans_total"].sum()
    assert abs(sum(df[b].sum() for b in bands) - total) < 1, "age bands do not sum to the total"
    assert 120_000 < total < 160_000, f"NYC veteran count {total} is outside a plausible range"


def test_empower_covers_the_five_boroughs() -> None:
    df = pl.read_parquet(REF / "empower_ny_zip.parquet").filter(pl.col("borough").is_not_null())
    assert set(df["borough"].unique()) == {
        "Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"}
    assert df["dme_power_dependent"].sum() > 30_000
    assert (df["dme_power_dependent"] >= 0).all()


def test_map_base_is_renderable() -> None:
    geo = json.loads((REF / "nyc_modzcta.geojson").read_text())
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) == 178
    assert all("modzcta" in f["properties"] for f in geo["features"])


# --------------------------------------------------------------------------- #
# Medication layer. Synthea emits RxNorm codes; RxNav maps them to the VA's own
# drug classes; med_climate_risk.csv attaches a mechanism and a weight to each.
# --------------------------------------------------------------------------- #

def test_med_climate_risk_crosswalk_is_well_formed() -> None:
    risk = pl.read_csv(REF / "med_climate_risk.csv")
    assert risk.height >= 45
    assert set(risk.columns) == {
        "va_class_id", "va_class_name", "mechanism", "hazard", "weight",
        "acb", "controlled", "cold_chain", "narrow_ti", "source"}
    assert risk["va_class_id"].n_unique() == risk.height, "duplicate VA class in the crosswalk"
    assert risk["weight"].min() > 0 and risk["weight"].max() <= 1.0
    assert risk["acb"].is_between(0, 3).all(), "ACB scale runs 0-3 per drug"
    assert set(risk["hazard"].unique()) <= {
        "heat", "outage", "breathing", "access_loss", "treatment_gap"}


def test_va_drug_classes_resolve_to_the_crosswalk() -> None:
    xw = pl.read_parquet(REF / "va_drug_class_members.parquet")
    risk = pl.read_csv(REF / "med_climate_risk.csv")
    assert xw.height > 3000, "RxNav membership pull looks truncated"
    assert set(xw["va_class_id"].unique()) <= set(risk["va_class_id"])
    # Every class in the crosswalk should have members except AH103, an obsolete
    # antihistamine class RxNav no longer populates.
    empty = set(risk["va_class_id"]) - set(xw["va_class_id"].unique())
    assert empty <= {"AH103"}, f"VA classes with no members: {empty}"
    assert xw["rxcui"].str.contains(r"^\d+$").all()


def test_the_cdc_named_heat_combination_is_representable() -> None:
    """CDC singles out ACE inhibitor or ARB *plus* a diuretic as additive heat risk.

    Both halves must be resolvable to real RxNorm codes, or the interaction term
    can never fire.
    """
    xw = pl.read_parquet(REF / "va_drug_class_members.parquet")
    diuretics = xw.filter(pl.col("va_class_id").is_in(
        ["CV700", "CV701", "CV702", "CV703", "CV704", "CV709"]))
    raas = xw.filter(pl.col("va_class_id").is_in(["CV800", "CV805"]))
    assert diuretics.height > 20, "no diuretic members"
    assert raas.height > 20, "no ACE inhibitor / ARB members"
    names = " ".join(xw["drug_name"].to_list()).lower()
    # The two most-prescribed drugs in the Synthea cohort, and the pair CDC names.
    assert "hydrochlorothiazide" in names
    assert "lisinopril" in names


def test_controlled_substances_are_flagged() -> None:
    """The VA disaster pharmacy benefit excludes controlled substances -- VA must fill
    them. That exclusion is what makes these veterans need an earlier, different action."""
    risk = pl.read_csv(REF / "med_climate_risk.csv")
    controlled = set(risk.filter(pl.col("controlled") == 1)["va_class_id"])
    assert {"CN101", "CN302", "CN802"} <= controlled, "opioids, benzos and stimulants must be flagged"


def test_cold_chain_medications_are_flagged() -> None:
    risk = pl.read_csv(REF / "med_climate_risk.csv")
    cold = risk.filter(pl.col("cold_chain") == 1)
    assert "HS501" in set(cold["va_class_id"]), "insulin must be flagged cold-chain"
    assert (cold.filter(pl.col("va_class_id") == "HS501")["hazard"] == "outage").all()

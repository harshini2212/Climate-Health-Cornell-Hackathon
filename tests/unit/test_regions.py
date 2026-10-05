"""Regions: the pipeline runs on a place that is not NYC, and NYC's old files still load.

The end-to-end test builds a made-up four-unit region (`tests/fixtures/regions/
toyville.yaml`) and runs cohort -> hazards -> score -> allocate on it with no NYC table in
reach of the geography. If any stage still names NYC -- a hard-coded borough list, a
reference path, a ZIP effect keyed to 178 MODZCTAs -- this is where it shows.
"""

from __future__ import annotations

import logging
import shutil
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from leeward import schema
from leeward.cohort import build
from leeward.decision import allocate
from leeward.geo import region
from leeward.ingest import hazards as hz
from leeward.model import score_prior

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "regions" / "toyville.yaml"

#: Unit code -> (lon, lat, subregion). Two units in each subregion.
UNITS = {"90001": (-1.00, 1.00, "North"), "90002": (-1.01, 1.02, "North"),
         "90003": (-1.00, 0.98, "South"), "90004": (-1.02, 0.97, "South")}


def _write_reference(ref: Path) -> None:
    """Every table toyville.yaml names, shaped like its NYC counterpart. Values invented."""
    ref.mkdir(parents=True, exist_ok=True)
    codes = list(UNITS)
    n = len(codes)
    pl.DataFrame({
        "toyzip": codes, "label": [f"Unit {c}" for c in codes], "zcta_members": codes,
        "pop_est": [10_000.0] * n,
        "lon": [UNITS[c][0] for c in codes], "lat": [UNITS[c][1] for c in codes],
    }).write_parquet(ref / "toy_units.parquet")

    bands = list(build.BANDS)
    vets = {f"vet_{s}_{b}": [40.0 + 5 * i for i in range(n)] for s in ("m", "f") for b in bands}
    pl.DataFrame({"zcta": codes, **vets, "pop_65plus": [2_000.0] * n,
                  "veterans_total": [sum(v[i] for v in vets.values()) for i in range(n)]}
                 ).write_parquet(ref / "toy_acs_veterans.parquet")

    cells = sorted({c for _, _, cols in build.RACE_ETHNICITY_CELLS.values() for c in cols})
    pl.DataFrame({"zcta": codes, **{c: [100.0] * n for c in cells}}
                 ).write_parquet(ref / "toy_acs_race.parquet")

    pl.DataFrame({"zip": codes, "borough": [UNITS[c][2] for c in codes],
                  "dme_power_dependent": [40] * n, "dme_oxygen": [20] * n,
                  "dme_esrd_dialysis": [10] * n}).write_parquet(ref / "toy_empower.parquet")

    pl.DataFrame({"zcta": codes, **{col: [12.0] * n for col in build.PLACES_RATES.values()}}
                 ).write_parquet(ref / "toy_places.parquet")
    pl.DataFrame({"zcta": codes, "hvi": [1, 3, 4, 5]}).write_parquet(ref / "toy_hvi.parquet")
    pl.DataFrame({"toyzip": codes, "evac_zone_min": [0, 1, 0, 2]}
                 ).write_parquet(ref / "toy_evac.parquet")
    pl.DataFrame({"toyzip": codes, "stormwater_flooded_frac": [0.0, 0.2, 0.05, 0.3]}
                 ).write_parquet(ref / "toy_stormwater.parquet")
    pl.DataFrame({
        "station_no": ["T01", "T02"], "name": ["North Clinic", "South Clinic"],
        "lat": [1.01, 0.97], "lon": [-1.0, -1.01], "borough": ["North", "South"],
        "evac_zone": pl.Series([0, 1], dtype=pl.Int8),
        "site_dependent_services": [True, False],
    }).write_parquet(ref / "toy_facilities.parquet")


@pytest.fixture(scope="module")
def toy(tmp_path_factory: pytest.TempPathFactory):
    home = tmp_path_factory.mktemp("toyville")
    shutil.copy(FIXTURE, home / FIXTURE.name)
    _write_reference(home / "reference")
    r = region.load_file(home / FIXTURE.name)
    yield r
    region.forget(r.id)


@pytest.fixture(scope="module")
def scenario() -> dict:
    return {
        "name": "toy_heat", "region": "toyville", "start_date": date(2026, 7, 1), "days": 14,
        "baseline": dict(hz.DEFAULT_BASELINE),
        "events": [
            {"day": 6, "type": "heat_wave", "heat_index_max_f": [98, 101, 103]},
            {"day": 7, "type": "outage", "frac": 0.6, "boroughs": ["South"],
             "duration_days": 2},
            {"day": 8, "type": "site_down", "facility": "T02", "duration_days": 3},
        ],
    }


def test_a_second_region_runs_cohort_to_actions_end_to_end(toy, scenario) -> None:
    cohort = build.build(300, seed=0, region=toy)
    assert set(cohort["region_id"]) == {"toyville"}
    assert set(cohort["geo_id"]) <= toy.geo_ids
    assert set(cohort["borough"]) <= set(toy.subregions)
    assert set(cohort["facility_id"]) <= {"T01", "T02"}

    hazards, sites = hz.assemble(hz.normalise(scenario))
    schema.validate(hazards, "hazards")
    assert set(hazards["region_id"]) == {"toyville"}
    assert set(hazards["geo_id"]) == toy.geo_ids
    south = hazards.filter(pl.col("borough") == "South")["geo_id"].unique().sort().to_list()
    assert south == ["90003", "90004"], "the scenario's borough selector reads the region"

    days = [scenario["start_date"] + timedelta(days=i) for i in range(5, 10)]
    scores = score_prior.score(cohort, hazards, sites, dates=days, n_draws=40)
    schema.validate(scores, "scores")
    assert set(scores["region_id"]) == {"toyville"}

    actions = allocate.allocate(scores, cohort, schema.DEFAULT_CAPACITY, {"borough": 0.2},
                                hazards=hazards, site_status=sites)
    schema.validate(actions, "actions")
    assert actions.height > 0
    assert set(actions["region_id"]) == {"toyville"}
    for (_, bucket), grp in actions.group_by("date", "capacity_bucket"):
        assert grp.height <= schema.DEFAULT_CAPACITY[bucket]


def test_cohort_rows_from_a_region_validate_against_that_regions_subregions(toy) -> None:
    cohort = build.build(50, seed=1, region=toy)
    wrong = cohort.with_columns(borough=pl.lit("Brooklyn"))
    with pytest.raises(schema.SchemaError, match="borough"):
        schema.validate(wrong, "cohort")


def test_allocate_refuses_a_panel_that_spans_two_regions(toy) -> None:
    """One capacity is one team's afternoon; it cannot be shared across two regions."""
    a = build.build(40, seed=0, region=toy)
    b = a.with_columns(region_id=pl.lit("nyc"))
    two = pl.concat([a, b.with_columns(pl.col("veteran_id") + "-B")])
    scores = pl.DataFrame(schema=schema.empty("scores").schema)
    with pytest.raises(ValueError, match="one region"):
        allocate.allocate(scores, two, schema.DEFAULT_CAPACITY)


# --------------------------------------------------------------------------- #
# The read-time alias: files written before regions existed
# --------------------------------------------------------------------------- #

def test_a_pre_region_parquet_still_loads(tmp_path, monkeypatch, caplog) -> None:
    """An old file has `modzcta` and no `region_id`. It reads as NYC with `geo_id`, loudly."""
    legacy = pl.DataFrame({
        "modzcta": ["10001", "10002"], "date": [date(2026, 7, 1)] * 2,
        **{c.name: [_sample(c)] * 2 for c in schema.TABLES["hazards"].columns
           if c.name not in ("region_id", "geo_id", "date")},
    })
    path = tmp_path / "hazards.parquet"
    legacy.write_parquet(path)
    monkeypatch.setattr(schema, "DATA", tmp_path)
    with caplog.at_level(logging.INFO, logger="leeward.schema"):
        df = schema.read("hazards")
    assert "modzcta" not in df.columns
    assert df["geo_id"].to_list() == ["10001", "10002"]
    assert set(df["region_id"]) == {region.DEFAULT}
    assert any("modzcta" in r.getMessage() for r in caplog.records), "aliasing must be visible"


def test_writers_never_emit_modzcta(tmp_path, monkeypatch) -> None:
    """The alias is read-only. A frame that still says modzcta is refused at write time."""
    assert all(c.name != "modzcta" for t in schema.TABLES.values() for c in t.columns)
    monkeypatch.setattr(schema, "DATA", tmp_path)
    df = schema.empty("hazards").with_columns(pl.lit("x").alias("modzcta"))
    with pytest.raises(schema.SchemaError, match="modzcta"):
        schema.write(df, "hazards")


def test_nyc_region_matches_what_used_to_be_hard_coded() -> None:
    nyc = region.get("nyc")
    assert nyc.subregions == ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")
    assert len(nyc.geo_ids) == 178
    assert nyc.county_fips["36061"] == "Manhattan"
    assert nyc.geojson.exists()
    assert [nyc.subregion_of(z) for z in ("10001", "10301", "10451", "11201", "11368")] == [
        "Manhattan", "Staten Island", "Bronx", "Brooklyn", "Queens"]


def test_a_region_id_cannot_name_a_path() -> None:
    with pytest.raises(region.RegionError):
        region.get("../data/reference/nyc")


def _sample(c: schema.Column):
    if c.dtype == pl.Boolean:
        return False
    if c.dtype.is_integer():
        return int(c.bounds[0]) if c.bounds else 0
    if c.dtype.is_float():
        return float(c.bounds[0]) if c.bounds else 0.0
    return c.values[0] if c.values else "x"


def test_two_regions_sharing_a_code_never_share_a_zip_effect(toy, tmp_path) -> None:
    """The simulator's ZIP effect is keyed by (region_id, geo_id), not by the code alone."""
    from leeward.cohort import simulate
    twin_yaml = toy.source.read_text().replace("id: toyville", "id: toytwin")
    twin_path = toy.source.with_name("toytwin.yaml")
    twin_path.write_text(twin_yaml)
    twin = region.load_file(twin_path)
    try:
        a = build.build(20, seed=0, region=toy)
        b = a.with_columns(region_id=pl.lit("toytwin"))
        truth = simulate.load()
        off_a = simulate.latent_offset(a, truth)
        off_b = simulate.latent_offset(b, truth)
        frailty_only = off_a - off_b      # same rows, same frailty: what is left is the ZIP
        assert (abs(frailty_only) > 1e-9).any(), "two regions got the same ZIP effects"
    finally:
        region.forget(twin.id)


def test_allocate_refuses_hazards_from_another_region(toy, scenario) -> None:
    """A hazard row from another region joins nothing, which would read as a calm day."""
    cohort = build.build(40, seed=0, region=toy)
    hazards, sites = hz.assemble(hz.normalise(scenario))
    days = [scenario["start_date"] + timedelta(days=6)]
    scores = score_prior.score(cohort, hazards, sites, dates=days, n_draws=20)
    with pytest.raises(ValueError, match="hazards from region"):
        allocate.allocate(scores, cohort, schema.DEFAULT_CAPACITY,
                          hazards=hazards.with_columns(region_id=pl.lit("nyc")),
                          site_status=sites)


def test_an_open_ended_range_honours_the_end_it_has() -> None:
    low = region.GeoRange(name="Low", hi=10500)
    assert low.holds(10001) and not low.holds(11201)
    high = region.GeoRange(name="High", lo=11000)
    assert high.holds(11201) and not high.holds(10001)

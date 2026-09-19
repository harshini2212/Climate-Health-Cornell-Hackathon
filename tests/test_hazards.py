"""Acceptance tests for the hazard assembler (lane `demo`).

Written before the module. What they pin down:

- every modzcta appears on every day of every scenario, with no nulls
- both tables satisfy the frozen contracts in leeward/schema.py
- the smoke scenario is a *replay* of the June 2023 AirNow monitors, not a simulation:
  the citywide max on 7 June 2023 must exceed 190 ug/m3 (the real peak was 203.5)
- the Sandy scenario marks station 630 down on day 63, because station 630 is the
  Manhattan VA and it sits in hurricane evacuation zone 1
- assembling twice gives the same bytes: the same click produces the same number
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from leeward import schema
from leeward.ingest import hazards

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
SCENARIO_NAMES = ["sandy_then_heat", "ida_flash_flood", "smoke_2023"]


@pytest.fixture(scope="module")
def built() -> dict[str, tuple[dict, pl.DataFrame, pl.DataFrame]]:
    out = {}
    for name in SCENARIO_NAMES:
        scen = hazards.load_scenario(SCENARIOS / f"{name}.yaml")
        hz, site = hazards.assemble(scen)
        out[name] = (scen, hz, site)
    return out


def _dates(hz: pl.DataFrame) -> list[date]:
    return sorted(hz["date"].unique().to_list())


# --------------------------------------------------------------------------- #
# Shape and contract
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_every_modzcta_appears_on_every_day(built, name) -> None:
    scen, hz, _ = built[name]
    real = pl.read_parquet(schema.REFERENCE / "nyc_modzcta.parquet")["modzcta"].to_list()
    dates = _dates(hz)
    assert len(dates) == scen["days"], f"{name}: expected {scen['days']} days, got {len(dates)}"
    assert hz.height == len(real) * scen["days"]
    per_day = hz.group_by("date").agg(pl.col("modzcta").n_unique().alias("n"))
    assert per_day["n"].min() == len(real) == per_day["n"].max()
    assert set(hz["modzcta"].unique().to_list()) == set(real)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_no_nulls_anywhere(built, name) -> None:
    _, hz, site = built[name]
    for df, label in ((hz, "hazards"), (site, "site_status")):
        nulls = {c: n for c, n in zip(df.columns, df.null_count().row(0), strict=True) if n}
        assert not nulls, f"{name}/{label} has nulls: {nulls}"


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_tables_match_their_contracts(built, name) -> None:
    _, hz, site = built[name]
    schema.validate(hz, "hazards")
    schema.validate(site, "site_status")


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_site_status_covers_every_facility_every_day(built, name) -> None:
    scen, _, site = built[name]
    fac = pl.read_parquet(schema.REFERENCE / "va_facilities_nyc_hazard.parquet")
    assert site.height == fac.height * scen["days"]
    assert set(site["facility_id"].unique().to_list()) == set(fac["station_no"].to_list())


def test_hot_day_is_the_82f_threshold(built) -> None:
    """NYC Health's 2026 mortality report puts the risk hinge at 82F. Not a tuning choice."""
    _, hz, _ = built["sandy_then_heat"]
    assert hz.filter(pl.col("hot_day") != (pl.col("heat_index_max_f") >= 82.0)).height == 0


def test_static_zip_joins_are_present(built) -> None:
    _, hz, _ = built["sandy_then_heat"]
    for col in ("evac_zone_min", "hvi", "stormwater_flooded_frac"):
        assert col in hz.columns, f"{col} is a static per-ZIP join and must be carried"
    ref = pl.read_parquet(schema.REFERENCE / "stormwater_by_modzcta.parquet")
    one_day = hz.filter(pl.col("date") == hz["date"].min()).select("modzcta", "stormwater_flooded_frac")
    joined = one_day.join(ref, on="modzcta", suffix="_ref")
    assert (joined["stormwater_flooded_frac"] - joined["stormwater_flooded_frac_ref"]).abs().max() < 1e-9


# --------------------------------------------------------------------------- #
# Replays, not simulations
# --------------------------------------------------------------------------- #

def test_smoke_peak_day_replays_the_real_monitors(built) -> None:
    _, hz, _ = built["smoke_2023"]
    peak = hz.filter(pl.col("date") == date(2023, 6, 7))
    assert peak.height > 0, "7 June 2023 must fall inside the smoke scenario window"
    assert peak["pm25"].max() > 190.0, "real 7 June 2023 peak was 203.5 ug/m3"
    assert peak["smoke_alert"].all(), "every ZIP was under a smoke alert on 7 June 2023"
    clear = hz.filter(pl.col("date") == date(2023, 6, 9))
    assert clear["pm25"].max() < 40.0, "the episode cleared by 9 June; 14.9 was the real max"
    # Off-episode days look like an ordinary NYC summer, not like smoke.
    ordinary = hz.filter(pl.col("date") < date(2023, 6, 1))
    assert ordinary["pm25"].mean() < 20.0
    assert not ordinary["smoke_alert"].any()


def test_sandy_marks_station_630_down_on_day_63(built) -> None:
    """Station 630 is the Manhattan VA campus, evac zone 1, the one that closed in 2012."""
    _, hz, site = built["sandy_then_heat"]
    dates = _dates(hz)
    s630 = site.filter(pl.col("facility_id") == "630").sort("date")
    assert s630["evac_zone"][0] == 1
    assert s630["site_dependent_services"][0], "630 runs dialysis, infusion and OTP"
    by_day = dict(zip(s630["date"], s630["site_down"], strict=True))
    assert by_day[dates[62]] is False, "630 is open the day before landfall"
    assert by_day[dates[63]] is True, "630 goes down on day 63"
    assert by_day[dates[63 + 44]] is True, "and stays down for 45 days"
    assert by_day[dates[63 + 45]] is False, "then reopens"
    others = site.filter((pl.col("facility_id") != "630") & pl.col("site_down"))
    assert others.height == 0, "only station 630 goes down in the Sandy scenario"


def test_sandy_surge_hits_the_evacuation_zones(built) -> None:
    _, hz, _ = built["sandy_then_heat"]
    dates = _dates(hz)
    landfall = hz.filter(pl.col("date") == dates[63])
    coastal = landfall.filter(pl.col("evac_zone_min").is_in([1, 2]))
    inland = landfall.filter(~pl.col("evac_zone_min").is_in([1, 2]))
    assert coastal.height > 20 and inland.height > 20
    assert coastal["flood_warning"].all() and coastal["surge_ft"].min() > 0
    assert not inland["flood_warning"].any() and inland["surge_ft"].max() == 0
    assert coastal["outage_frac"].min() >= 0.8
    assert coastal["mail_delivery_disrupted"].all()
    assert landfall["evac_zone_ordered"].max() == 2
    quiet = hz.filter(pl.col("date") == dates[30])
    assert not quiet["flood_warning"].any() and quiet["outage_frac"].max() < 0.05
    assert quiet["evac_zone_ordered"].max() == 0


def test_sandy_heat_wave_follows_the_storm(built) -> None:
    _, hz, _ = built["sandy_then_heat"]
    dates = _dates(hz)
    for i, expected in zip((66, 67, 68), (96, 99, 97), strict=True):
        day = hz.filter(pl.col("date") == dates[i])
        assert day["heat_alert"].all() and day["hot_day"].all()
        assert abs(day["heat_index_max_f"].mean() - expected) < 3.0
    assert not hz.filter(pl.col("date") == dates[65])["heat_alert"].any()


def test_ida_flash_flood_trips_floodnet_where_the_stormwater_map_says(built) -> None:
    _, hz, _ = built["ida_flash_flood"]
    ida = hz.filter(pl.col("date") == date(2021, 9, 1))
    assert ida.height > 0, "1 September 2021 must fall inside the Ida window"
    assert ida["flash_flood_emergency"].any()
    tripped = ida.filter(pl.col("floodnet_trip"))
    untripped = ida.filter(~pl.col("floodnet_trip"))
    assert tripped.height > 10
    assert tripped["stormwater_flooded_frac"].min() > untripped["stormwater_flooded_frac"].min()
    assert tripped["mail_delivery_disrupted"].all()
    before = hz.filter(pl.col("date") == date(2021, 8, 20))
    assert not before["flash_flood_emergency"].any() and not before["floodnet_trip"].any()


# --------------------------------------------------------------------------- #
# Determinism and the CLI
# --------------------------------------------------------------------------- #

def test_assembly_is_deterministic() -> None:
    scen = hazards.load_scenario(SCENARIOS / "sandy_then_heat.yaml")
    a_hz, a_site = hazards.assemble(scen)
    b_hz, b_site = hazards.assemble(scen)
    assert a_hz.equals(b_hz) and a_site.equals(b_site)


def test_cli_writes_both_tables_through_schema_write(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(schema, "DATA", tmp_path)
    rc = hazards.main(["--scenario", str(SCENARIOS / "smoke_2023.yaml")])
    assert rc == 0
    hz = pl.read_parquet(tmp_path / "hazards.parquet")
    site = pl.read_parquet(tmp_path / "site_status.parquet")
    schema.validate(hz, "hazards")
    schema.validate(site, "site_status")


def test_module_makes_no_network_calls() -> None:
    src = Path(hazards.__file__).read_text(encoding="utf-8")
    for bad in ("requests.", "urllib", "httpx", "urlopen"):
        assert bad not in src, f"hazards.py must read data/reference only; found {bad!r}"

"""Assemble hazards.parquet and site_status.parquet from a scenario YAML plus data/reference.

    python -m leeward.ingest.hazards --scenario scenarios/sandy_then_heat.yaml

Reads only `data/reference/`. Never touches the network. Everything that is random is
seeded from the scenario, so the same YAML always produces the same bytes.

Scenario YAML
-------------
    name: sandy_then_heat
    start_date: 2026-06-01        # day 0
    days: 120
    baseline:                     # the ordinary summer that the events sit on top of
      seed: 0
      heat_index_mean_f: 70       # annual mean of daily max heat index
      heat_index_amp_f: 16        # seasonal amplitude, peak on 24 July
      heat_index_noise_f: 4       # day-to-day citywide noise
      pm25_mean: 9.0
    events:
      - {day: 63, type: site_down, facility: "630", duration_days: 45}
      ...

Every event has `day` (0-indexed), an optional `duration_days` (default 1), and an optional
ZIP selector: any of `zones` (evac_zone_min in the list), `boroughs`, `modzctas`,
`stormwater_min_frac`. Selector terms are ANDed; no selector means every ZIP.

Event types:
    flood_watch              flood_watch=True on selected ZIPs
    coastal_flood_warning    flood_warning=True, surge_ft, evacuate_zones -> evac_zone_ordered
    flash_flood_emergency    flash_flood_emergency + flood_warning; floodnet_trip where the
                             stormwater map says the ZIP floods (floodnet_min_frac, default 0.10)
    outage                   outage_frac = frac on selected ZIPs
    heat_wave                heat_index_max_f: [..per day..], heat_alert: true
    smoke_replay             real AirNow PM2.5 by nearest monitor, replay_start -> day
    site_down                facility (station number) closed for duration_days

Two things are replays, not simulations: the June 2023 smoke episode comes from the AirNow
monitors in `airnow_pm25_nyc_smoke2023.parquet`, and the Sandy closure is station 630
because that is the Manhattan VA and it sits in evacuation zone 1.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from leeward import schema

HOT_DAY_F = 82.0            # NYC Health 2026 heat mortality report: "non-extreme hot days"
SMOKE_ALERT_UGM3 = 55.5     # PM2.5 24-hr concentration at which AQI crosses 150, "unhealthy"
MAIL_OUTAGE_FRAC = 0.5      # a ZIP more than half dark does not get mail delivered
PEAK_DOY = 205              # 24 July, NYC's climatological heat-index peak


# --------------------------------------------------------------------------- #
# Reference tables
# --------------------------------------------------------------------------- #

def _ref(name: str) -> pl.DataFrame:
    path = schema.REFERENCE / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/fetch_sources.py")
    return pl.read_parquet(path)


def _borough(modzcta: str) -> str:
    """NYC ZIP prefix ranges. MODZCTA carries no borough column."""
    n = int(modzcta)
    if 10001 <= n <= 10282:
        return "Manhattan"
    if 10301 <= n <= 10314:
        return "Staten Island"
    if 10451 <= n <= 10475:
        return "Bronx"
    if 11201 <= n <= 11256:
        return "Brooklyn"
    return "Queens"


def zip_frame() -> pl.DataFrame:
    """The 178 MODZCTAs with their static hazard exposure, sorted by code."""
    mz = _ref("nyc_modzcta").select("modzcta", "lon", "lat")
    evac = _ref("evac_zone_by_modzcta").select("modzcta", "evac_zone_min")
    storm = _ref("stormwater_by_modzcta")
    hvi = _ref("hvi_by_zcta").rename({"zcta": "modzcta"})
    return (mz.join(evac, on="modzcta", how="left")
              .join(storm, on="modzcta", how="left")
              .join(hvi, on="modzcta", how="left")
              .with_columns(
                  pl.col("modzcta").map_elements(_borough, return_dtype=pl.Utf8).alias("borough"),
                  pl.col("evac_zone_min").fill_null(0).cast(pl.Int32),
                  pl.col("stormwater_flooded_frac").fill_null(0.0).cast(pl.Float64),
                  pl.col("hvi").fill_null(3).cast(pl.Int32),
              )
              .sort("modzcta"))


def facility_frame() -> pl.DataFrame:
    return (_ref("va_facilities_nyc_hazard")
            .select("station_no", "name", "lat", "lon", "evac_zone", "site_dependent_services")
            .with_columns(pl.col("evac_zone").fill_null(0).cast(pl.Int32))
            .sort("station_no"))


# --------------------------------------------------------------------------- #
# Scenario
# --------------------------------------------------------------------------- #

DEFAULT_BASELINE: dict[str, Any] = {
    "seed": 0,
    "heat_index_mean_f": 70.0,
    "heat_index_amp_f": 16.0,
    "heat_index_noise_f": 4.0,
    "pm25_mean": 9.0,
}


def load_scenario(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        scen = yaml.safe_load(fh)
    for key in ("name", "start_date", "days"):
        if key not in scen:
            raise ValueError(f"{path}: scenario needs '{key}'")
    start = scen["start_date"]
    scen["start_date"] = start if isinstance(start, date) else date.fromisoformat(str(start))
    scen["days"] = int(scen["days"])
    scen["baseline"] = {**DEFAULT_BASELINE, **(scen.get("baseline") or {})}
    scen["events"] = list(scen.get("events") or [])
    for ev in scen["events"]:
        if "day" not in ev or "type" not in ev:
            raise ValueError(f"{path}: every event needs 'day' and 'type': {ev}")
    return scen


def select_zips(zips: pl.DataFrame, ev: dict[str, Any]) -> np.ndarray:
    """Boolean mask over `zips` rows. Selector terms are ANDed; none means everything."""
    mask = np.ones(zips.height, dtype=bool)
    if "zones" in ev:
        mask &= zips["evac_zone_min"].is_in([int(z) for z in ev["zones"]]).to_numpy()
    if "boroughs" in ev:
        mask &= zips["borough"].is_in(list(ev["boroughs"])).to_numpy()
    if "modzctas" in ev:
        mask &= zips["modzcta"].is_in([str(z) for z in ev["modzctas"]]).to_numpy()
    if "stormwater_min_frac" in ev:
        mask &= (zips["stormwater_flooded_frac"] >= float(ev["stormwater_min_frac"])).to_numpy()
    return mask


def _event_days(ev: dict[str, Any], n_days: int) -> range:
    start = int(ev["day"])
    dur = int(ev.get("duration_days", 1))
    return range(max(start, 0), min(start + dur, n_days))


# --------------------------------------------------------------------------- #
# Baseline weather
# --------------------------------------------------------------------------- #

def _baseline_heat_index(dates: list[date], zips: pl.DataFrame, bl: dict[str, Any],
                         rng: np.random.Generator) -> np.ndarray:
    """(T, Z) daily max heat index: seasonal cosine + citywide noise + a small HVI offset.

    The HVI offset is the urban-heat-island signal: the index is built partly from surface
    temperature, so a band-5 ZIP runs a couple of degrees hotter than a band-1 ZIP.
    """
    doy = np.array([d.timetuple().tm_yday for d in dates], dtype=float)
    seasonal = bl["heat_index_mean_f"] + bl["heat_index_amp_f"] * np.cos(
        2 * np.pi * (doy - PEAK_DOY) / 365.25)
    noise = rng.normal(0.0, bl["heat_index_noise_f"], size=len(dates))
    hvi_offset = (zips["hvi"].to_numpy().astype(float) - 3.0) * 0.8
    return (seasonal + noise)[:, None] + hvi_offset[None, :]


def _baseline_pm25(n_days: int, n_zips: int, bl: dict[str, Any],
                   rng: np.random.Generator) -> np.ndarray:
    """(T, Z) ordinary-summer PM2.5: a citywide daily level with mild ZIP-level scatter."""
    mean = float(bl["pm25_mean"])
    city = rng.gamma(shape=4.0, scale=mean / 4.0, size=n_days)
    scatter = rng.normal(1.0, 0.08, size=(n_days, n_zips))
    return np.clip(city[:, None] * scatter, 0.5, None)


# --------------------------------------------------------------------------- #
# Smoke replay
# --------------------------------------------------------------------------- #

def _nearest_monitor_pm25(zips: pl.DataFrame, monitors: pl.DataFrame) -> np.ndarray:
    """PM2.5 per ZIP from the nearest reporting monitor (equirectangular distance)."""
    zlat = np.radians(zips["lat"].to_numpy())
    zlon = np.radians(zips["lon"].to_numpy())
    mlat = np.radians(monitors["lat"].to_numpy())
    mlon = np.radians(monitors["lon"].to_numpy())
    dx = (zlon[:, None] - mlon[None, :]) * np.cos((zlat[:, None] + mlat[None, :]) / 2)
    dy = zlat[:, None] - mlat[None, :]
    nearest = np.argmin(dx * dx + dy * dy, axis=1)
    return monitors["value"].to_numpy()[nearest]


def _apply_smoke_replay(pm25: np.ndarray, zips: pl.DataFrame, dates: list[date],
                        ev: dict[str, Any]) -> None:
    source = ev.get("source", "airnow_pm25_nyc_smoke2023")
    mon = (_ref(source)
           .filter(pl.col("parameter").str.starts_with("PM2.5"))
           .with_columns(pl.col("date").str.strptime(pl.Date, "%m/%d/%y").alias("obs_date"))
           .drop_nulls(["lat", "lon", "value"]))
    replay_start = ev.get("replay_start")
    if replay_start is None:
        replay_start = mon["obs_date"].min()
    elif not isinstance(replay_start, date):
        replay_start = date.fromisoformat(str(replay_start))
    day0 = int(ev["day"])
    for obs_date, grp in mon.group_by("obs_date", maintain_order=True):
        obs = obs_date[0] if isinstance(obs_date, tuple) else obs_date
        t = day0 + (obs - replay_start).days
        if 0 <= t < len(dates):
            pm25[t, :] = _nearest_monitor_pm25(zips, grp)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def assemble(scen: dict[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (hazards, site_status) for the scenario. Pure: reads reference tables only."""
    zips = zip_frame()
    fac = facility_frame()
    n_days, n_zips = scen["days"], zips.height
    dates = [scen["start_date"] + timedelta(days=i) for i in range(n_days)]
    bl = scen["baseline"]
    rng = np.random.default_rng(int(bl["seed"]))

    heat = _baseline_heat_index(dates, zips, bl, rng)
    pm25 = _baseline_pm25(n_days, n_zips, bl, rng)
    heat_alert = np.zeros(n_days, dtype=bool)
    flood_watch = np.zeros((n_days, n_zips), dtype=bool)
    flood_warning = np.zeros((n_days, n_zips), dtype=bool)
    flash = np.zeros((n_days, n_zips), dtype=bool)
    surge = np.zeros((n_days, n_zips), dtype=float)
    evac_ordered = np.zeros(n_days, dtype=np.int32)
    floodnet = np.zeros((n_days, n_zips), dtype=bool)
    outage = np.zeros((n_days, n_zips), dtype=float)
    site_down = np.zeros((n_days, fac.height), dtype=bool)

    storm_frac = zips["stormwater_flooded_frac"].to_numpy()
    zone_min = zips["evac_zone_min"].to_numpy()
    stations = fac["station_no"].to_list()

    for ev in scen["events"]:
        kind = ev["type"]
        days = _event_days(ev, n_days)
        sel = select_zips(zips, ev)

        if kind == "flood_watch":
            for t in days:
                flood_watch[t, sel] = True

        elif kind == "coastal_flood_warning":
            zones = [int(z) for z in ev.get("evacuate_zones", ev.get("zones", []))]
            for t in days:
                flood_warning[t, sel] = True
                surge[t, sel] = np.maximum(surge[t, sel], float(ev.get("surge_ft", 0.0)))
                floodnet[t, sel & (storm_frac >= float(ev.get("floodnet_min_frac", 0.15)))] = True
                if zones:
                    evac_ordered[t] = max(evac_ordered[t], max(zones))

        elif kind == "flash_flood_emergency":
            for t in days:
                flash[t, sel] = True
                flood_warning[t, sel] = True
                floodnet[t, sel & (storm_frac >= float(ev.get("floodnet_min_frac", 0.10)))] = True

        elif kind == "outage":
            for t in days:
                outage[t, sel] = np.maximum(outage[t, sel], float(ev["frac"]))

        elif kind == "heat_wave":
            values = ev.get("heat_index_max_f")
            if values is None:
                raise ValueError("heat_wave needs heat_index_max_f: [per-day values]")
            values = [float(v) for v in (values if isinstance(values, list) else [values])]
            hvi_offset = (zips["hvi"].to_numpy().astype(float) - 3.0) * 0.8
            for i, v in enumerate(values):
                t = int(ev["day"]) + i
                if 0 <= t < n_days:
                    heat[t, :] = v + hvi_offset
                    heat_alert[t] = bool(ev.get("heat_alert", True))

        elif kind == "smoke_replay":
            _apply_smoke_replay(pm25, zips, dates, ev)

        elif kind == "site_down":
            station = str(ev["facility"])
            if station not in stations:
                raise ValueError(f"site_down: unknown station {station!r}; "
                                 f"see va_facilities_nyc_hazard.parquet")
            j = stations.index(station)
            for t in days:
                site_down[t, j] = True

        else:
            raise ValueError(f"unknown event type {kind!r}")

    heat = np.round(heat, 1)
    pm25 = np.round(pm25, 1)
    ordered_here = (zone_min[None, :] > 0) & (zone_min[None, :] <= evac_ordered[:, None])
    mail_disrupted = (flood_warning | flash | (outage >= MAIL_OUTAGE_FRAC) | ordered_here)

    modz = zips["modzcta"].to_list()
    hazards = pl.DataFrame({
        "modzcta": np.tile(np.array(modz, dtype=object), n_days).tolist(),
        "date": pl.Series([d for d in dates for _ in range(n_zips)], dtype=pl.Date),
        "heat_index_max_f": heat.ravel(),
        "hot_day": (heat >= HOT_DAY_F).ravel(),
        "heat_alert": np.repeat(heat_alert, n_zips),
        "pm25": pm25.ravel(),
        "smoke_alert": (pm25 >= SMOKE_ALERT_UGM3).ravel(),
        "flood_watch": flood_watch.ravel(),
        "flood_warning": flood_warning.ravel(),
        "flash_flood_emergency": flash.ravel(),
        "surge_ft": surge.ravel(),
        "evac_zone_ordered": np.repeat(evac_ordered, n_zips).astype(np.int32),
        "floodnet_trip": floodnet.ravel(),
        "stormwater_flooded_frac": np.tile(storm_frac, n_days),
        "outage_frac": np.round(outage.ravel(), 2),
        "mail_delivery_disrupted": mail_disrupted.ravel(),
        # static per-ZIP joins the model and the map both want
        "evac_zone_min": np.tile(zone_min, n_days).astype(np.int32),
        "hvi": np.tile(zips["hvi"].to_numpy(), n_days).astype(np.int32),
        "borough": np.tile(np.array(zips["borough"].to_list(), dtype=object), n_days).tolist(),
    })

    n_fac = fac.height
    site_status = pl.DataFrame({
        "facility_id": np.tile(np.array(stations, dtype=object), n_days).tolist(),
        "date": pl.Series([d for d in dates for _ in range(n_fac)], dtype=pl.Date),
        "site_down": site_down.ravel(),
        "evac_zone": np.tile(fac["evac_zone"].to_numpy(), n_days).astype(np.int32),
        "site_dependent_services": np.tile(fac["site_dependent_services"].to_numpy(), n_days),
    })
    return hazards, site_status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--scenario", required=True, help="path to a scenario YAML")
    args = ap.parse_args(argv)

    scen = load_scenario(args.scenario)
    hazards, site_status = assemble(scen)
    p1 = schema.write(hazards, "hazards")
    p2 = schema.write(site_status, "site_status")
    downs = site_status.filter(pl.col("site_down")).group_by("facility_id").len()
    print(f"scenario {scen['name']}: {scen['days']} days from {scen['start_date']}")
    print(f"  {p1.name:22s} {hazards.height:>8,} rows")
    print(f"  {p2.name:22s} {site_status.height:>8,} rows"
          + (f"  ({', '.join(f'{r[0]} down {r[1]}d' for r in downs.iter_rows())})"
             if downs.height else ""))
    print(f"  generated {datetime.now():%Y-%m-%d %H:%M}, no network")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

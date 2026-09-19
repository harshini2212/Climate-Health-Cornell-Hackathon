#!/usr/bin/env python3
"""Vendor every Leeward data source into the repo.

Run once before the hackathon so that hour 0 is not spent on network archaeology:

    python scripts/fetch_sources.py            # everything that is small and keyless
    python scripts/fetch_sources.py --only hvi empower
    python scripts/fetch_sources.py --heavy    # also the 4 GB VA Synthea release

Design rules
------------
* Every fetcher is keyless unless it prints a KEY note. Nothing here needs a signup
  to produce a working demo.
* Each fetcher writes ONE parquet (or geojson) into ``data/reference/`` and one row
  into ``data/reference/manifest.json`` recording url, sha256, rows and fetch time.
* ``data/raw/`` holds the big intermediate downloads and is gitignored.
  ``data/reference/`` is small and IS committed, so ``make demo`` works offline
  from a clean clone.
* A fetcher that fails prints a warning and leaves any existing snapshot alone.
  Nothing in the build is allowed to depend on the network at demo time.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import traceback
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REF = ROOT / "data" / "reference"
MANIFEST = REF / "manifest.json"

UA = {"User-Agent": "leeward-hackathon/0.1 (health-in-climate-ai NYC 2026; synthetic data only)"}
TIMEOUT = 120

# NYC counties, for filtering state-wide sources down to the five boroughs.
NYC_COUNTY_FIPS = {"36005": "Bronx", "36047": "Brooklyn", "36061": "Manhattan",
                   "36081": "Queens", "36085": "Staten Island"}
NYC_COUNTY_NAMES = {"BRONX": "Bronx", "KINGS": "Brooklyn", "NEW YORK": "Manhattan",
                    "QUEENS": "Queens", "RICHMOND": "Staten Island"}

REGISTRY: dict[str, "Fetcher"] = {}


class Fetcher:
    def __init__(self, name, url, note, fn, heavy=False):
        self.name, self.url, self.note, self.fn, self.heavy = name, url, note, fn, heavy


def source(name: str, url: str, note: str, heavy: bool = False):
    def deco(fn):
        REGISTRY[name] = Fetcher(name, url, note, fn, heavy)
        return fn
    return deco


def get(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers=UA, timeout=TIMEOUT, **kw)
    r.raise_for_status()
    return r


def socrata_rows(domain: str, dataset: str, limit: int = 200_000, **params) -> list[dict]:
    """Page a Socrata resource. Scalars come back as strings; geometry as nested dicts."""
    rows, offset = [], 0
    while True:
        p = {"$limit": 50_000, "$offset": offset, **params}
        batch = get(f"https://{domain}/resource/{dataset}.json", params=p).json()
        rows.extend(batch)
        if len(batch) < 50_000 or len(rows) >= limit:
            break
        offset += 50_000
    return rows


def socrata(domain: str, dataset: str, limit: int = 200_000, **params) -> pl.DataFrame:
    """Same, as a DataFrame, with geometry columns dropped so polars stays happy."""
    rows = socrata_rows(domain, dataset, limit, **params)
    flat = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]
    return pl.DataFrame(flat, infer_schema_length=None, strict=False)


def arcgis(service: str, layer: int, where: str = "1=1", out_fields: str = "*",
           geometry: bool = False) -> list[dict]:
    """Page an ArcGIS FeatureServer layer, respecting maxRecordCount."""
    feats, offset = [], 0
    while True:
        p = {"where": where, "outFields": out_fields, "returnGeometry": str(geometry).lower(),
             "f": "geojson" if geometry else "json", "resultOffset": offset,
             "resultRecordCount": 2000, "outSR": 4326}
        js = get(f"{service}/{layer}/query", params=p).json()
        if "error" in js:
            raise RuntimeError(f"ArcGIS {service}/{layer}: {js['error']}")
        got = js.get("features", [])
        feats.extend(got)
        if len(got) < 2000:
            break
        offset += 2000
    return feats


def write(df: pl.DataFrame, name: str, url: str) -> dict:
    REF.mkdir(parents=True, exist_ok=True)
    path = REF / f"{name}.parquet"
    df.write_parquet(path)
    return {"file": path.name, "url": url, "rows": df.height, "cols": df.width,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            "bytes": path.stat().st_size,
            "fetched_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}


# --------------------------------------------------------------------------- #
# Geometry base: NYC's own ZCTA polygons. Small, official, and carries the
# MODZCTA -> member-ZCTA list we need to attach ACS and CDC tables.
# --------------------------------------------------------------------------- #

@source("modzcta", "https://data.cityofnewyork.us/resource/pri4-ifjk.json",
        "NYC Modified ZIP Code Tabulation Areas: the map base and the spatial-join unit.")
def fetch_modzcta():
    import geopandas as gpd
    from shapely.geometry import shape

    raw = socrata_rows("data.cityofnewyork.us", "pri4-ifjk")
    recs = []
    for r in raw:
        geom = r.get("the_geom")
        if not geom:
            continue
        recs.append({"modzcta": str(r.get("modzcta")), "label": r.get("label"),
                     "zcta_members": r.get("zcta"), "pop_est": r.get("pop_est"),
                     "geometry": shape(geom)})
    gdf = gpd.GeoDataFrame(recs, crs="EPSG:4326")
    # Centroids in a projected CRS so distances and areas are in metres.
    proj = gdf.to_crs(2263)  # NY State Plane Long Island (US ft)
    cent = proj.geometry.centroid.to_crs(4326)
    gdf["lon"], gdf["lat"] = cent.x, cent.y
    gdf["area_sqft"] = proj.geometry.area

    (REF / "nyc_modzcta.geojson").write_text(gdf.to_json())
    df = pl.DataFrame(gdf.drop(columns="geometry").to_dict("list")).with_columns(
        pl.col("pop_est").cast(pl.Float64, strict=False))
    return write(df, "nyc_modzcta", "https://data.cityofnewyork.us/d/pri4-ifjk")


@source("hvi", "https://data.cityofnewyork.us/resource/4mhf-duep.json",
        "NYC Heat Vulnerability Index 1-5, published per ZCTA20. The ZIP heat prior.")
def fetch_hvi():
    df = socrata("data.cityofnewyork.us", "4mhf-duep").select(
        pl.col("zcta20").alias("zcta"), pl.col("hvi").cast(pl.Int8))
    return write(df, "hvi_by_zcta", "https://data.cityofnewyork.us/d/4mhf-duep")


@source("empower", "https://services2.arcgis.com/ZQ4jTQn6k7VPXEwO/arcgis/rest/services/"
        "HHS_emPOWER_REST_Service_Public/FeatureServer",
        "HHS emPOWER: electricity-dependent Medicare beneficiaries per ZIP. "
        "Layer 1 = all DME, and it carries the O2 and ESRD-dialysis columns too.")
def fetch_empower():
    svc = ("https://services2.arcgis.com/ZQ4jTQn6k7VPXEwO/arcgis/rest/services/"
           "HHS_emPOWER_REST_Service_Public/FeatureServer")
    feats = arcgis(svc, 1, where="STATE='NY'",
                   out_fields=("STATE,COUNTY,Zip_Code,Power_Dependent_Devices_DME,"
                               "Facility_ESRD_Dialysis_Any_DME,O2_Services_Any_DME,"
                               "Home_Health_Services_Any_DME,AtHome_Hospice_Any_DME,"
                               "Any_Healthcare_Srvc_Any_DME"))
    df = pl.DataFrame([f["attributes"] for f in feats]).rename({
        "Zip_Code": "zip", "COUNTY": "county",
        "Power_Dependent_Devices_DME": "dme_power_dependent",
        "Facility_ESRD_Dialysis_Any_DME": "dme_esrd_dialysis",
        "O2_Services_Any_DME": "dme_oxygen",
        "Home_Health_Services_Any_DME": "dme_home_health",
        "AtHome_Hospice_Any_DME": "dme_hospice",
        "Any_Healthcare_Srvc_Any_DME": "dme_any_service"})
    df = df.with_columns(
        pl.col("county").str.to_uppercase().replace_strict(NYC_COUNTY_NAMES, default=None)
          .alias("borough"))
    return write(df, "empower_ny_zip", svc)


@source("places", "https://data.cdc.gov/resource/kee5-23sr.json",
        "CDC PLACES 2025, ZCTA level. Supplies the augment priors that were previously "
        "invented: mobility, self-care, independent living, loneliness, emotional support, "
        "utility shutoff, transport barrier, plus COPD/asthma/cancer/depression prevalence.")
def fetch_places():
    keep = ["zcta5", "totalpopulation", "totalpop18plus",
            "copd_crudeprev", "casthma_crudeprev", "cancer_crudeprev", "chd_crudeprev",
            "diabetes_crudeprev", "depression_crudeprev", "mhlth_crudeprev",
            "mobility_crudeprev", "selfcare_crudeprev", "indeplive_crudeprev",
            "disability_crudeprev", "cognition_crudeprev",
            "loneliness_crudeprev", "emotionspt_crudeprev",
            "shututility_crudeprev", "lacktrpt_crudeprev",
            "housinsecu_crudeprev", "foodinsecu_crudeprev", "foodstamp_crudeprev",
            "access2_crudeprev", "ghlth_crudeprev"]
    # Pull only the ZCTAs NYC cares about; PLACES has 32k nationally.
    zctas = _nyc_zctas()
    where = "zcta5 in ({})".format(",".join(f"'{z}'" for z in sorted(zctas)))
    df = socrata("data.cdc.gov", "kee5-23sr", **{"$select": ",".join(keep), "$where": where})
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False)
                          for c in df.columns if c != "zcta5"]).rename({"zcta5": "zcta"})
    return write(df, "places_zcta_nyc", "https://data.cdc.gov/d/kee5-23sr")


@source("svi", "https://svi.cdc.gov/Documents/Data/2022/csv/states/NewYork.csv",
        "CDC/ATSDR Social Vulnerability Index 2022, NY census tracts. "
        "Feeds the ZIP random-effect prior and the fairness strata.")
def fetch_svi():
    txt = get("https://svi.cdc.gov/Documents/Data/2022/csv/states/NewYork.csv").text
    df = pl.read_csv(io.StringIO(txt), infer_schema_length=0)
    keep = [c for c in df.columns if c in {
        "ST", "STATE", "ST_ABBR", "COUNTY", "FIPS", "LOCATION", "E_TOTPOP", "E_HU", "E_HH",
        "RPL_THEME1", "RPL_THEME2", "RPL_THEME3", "RPL_THEME4", "RPL_THEMES",
        "EP_POV150", "EP_UNEMP", "EP_NOHSDP", "EP_AGE65", "EP_DISABL", "EP_SNGPNT",
        "EP_MINRTY", "EP_LIMENG", "EP_MUNIT", "EP_MOBILE", "EP_CROWD", "EP_NOVEH",
        "EP_GROUPQ", "EP_UNINSUR"}]
    df = df.select(keep).with_columns(pl.col("FIPS").str.slice(0, 5).alias("county_fips"))
    df = df.filter(pl.col("county_fips").is_in(list(NYC_COUNTY_FIPS)))
    df = df.with_columns(pl.col("county_fips").replace_strict(NYC_COUNTY_FIPS, default=None)
                           .alias("borough"))
    num = [c for c in df.columns if c.startswith(("E_", "EP_", "RPL_"))]
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num])
    return write(df, "svi_nyc_tract", "https://svi.cdc.gov/Documents/Data/2022/csv/states/NewYork.csv")


@source("nri", "https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/"
        "National_Risk_Index_Census_Tracts/FeatureServer",
        "FEMA National Risk Index, census tracts. Long-run heat-wave, hurricane, "
        "coastal-flood and riverine-flood hazard priors. "
        "NOTE: the old hazards.fema.gov static zip now 301s to a landing page; "
        "this FeatureServer is the live path.")
def fetch_nri():
    svc = ("https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/"
           "National_Risk_Index_Census_Tracts/FeatureServer")
    # NRI names the heat-wave family HWAV_*, not HRWV_*.
    fields = ("STCOFIPS,TRACTFIPS,STATEABBRV,COUNTY,POPULATION,RISK_SCORE,RISK_RATNG,"
              "SOVI_SCORE,RESL_SCORE,HWAV_AFREQ,HWAV_EALT,HWAV_RISKS,"
              "HRCN_AFREQ,HRCN_EALT,HRCN_RISKS,CFLD_AFREQ,CFLD_EALT,CFLD_RISKS")
    where = "STCOFIPS IN ({})".format(",".join(f"'{c}'" for c in NYC_COUNTY_FIPS))
    feats = arcgis(svc, 0, where=where, out_fields=fields)
    df = pl.DataFrame([f["attributes"] for f in feats])
    df = df.with_columns(pl.col("STCOFIPS").replace_strict(NYC_COUNTY_FIPS, default=None)
                           .alias("borough"))
    return write(df, "fema_nri_nyc_tract", svc)


@source("evac", "https://data.cityofnewyork.us/resource/epne-qv9x.json",
        "NYC Hurricane Evacuation Zones 1-6, area-weighted onto MODZCTA. "
        "This is the surge-exposure term and the join that would otherwise eat an "
        "hour of hackathon time.")
def fetch_evac():
    import geopandas as gpd
    from shapely.geometry import shape

    raw = socrata_rows("data.cityofnewyork.us", "epne-qv9x")
    # The layer carries zones 1-7 plus a polygon coded "X" for everywhere that is
    # not in any evacuation zone. X is dropped; it is the complement, not a zone.
    zones = gpd.GeoDataFrame(
        [{"evac_zone": int(r["hurricane_"]), "geometry": shape(r["the_geom"])}
         for r in raw if r.get("the_geom") and str(r.get("hurricane_", "")).isdigit()],
        crs="EPSG:4326").to_crs(2263)
    zones["geometry"] = zones.geometry.buffer(0)

    zcta = gpd.read_file(REF / "nyc_modzcta.geojson").to_crs(2263)
    zcta["geometry"] = zcta.geometry.buffer(0)
    zcta["zcta_area"] = zcta.geometry.area

    inter = gpd.overlay(zcta[["modzcta", "zcta_area", "geometry"]], zones, how="intersection")
    inter["frac"] = inter.geometry.area / inter["zcta_area"]

    rows = []
    for mz, grp in inter.groupby("modzcta"):
        by_zone = grp.groupby("evac_zone")["frac"].sum().to_dict()
        covered = sum(by_zone.values())
        # The operational number: the most protective (lowest-numbered) zone that
        # covers a meaningful share, plus the share of the ZIP in any zone.
        worst = min((z for z, f in by_zone.items() if f >= 0.02), default=0)
        rows.append({"modzcta": mz, "evac_zone_min": int(worst),
                     "evac_frac_any": float(min(covered, 1.0)),
                     **{f"evac_frac_z{z}": float(by_zone.get(z, 0.0)) for z in range(1, 8)}})
    df = pl.DataFrame(rows)
    # ZIPs with no intersection at all are zone 0, fully inland.
    all_mz = pl.read_parquet(REF / "nyc_modzcta.parquet").select("modzcta")
    df = all_mz.join(df, on="modzcta", how="left").fill_null(0)
    return write(df, "evac_zone_by_modzcta", "https://data.cityofnewyork.us/d/epne-qv9x")


@source("stormwater", "https://data.cityofnewyork.us/api/views/9i7c-xyvv/files/"
        "6ce7b252-a38c-47ae-a823-680f443227e5?filename=NYCFloodStormwaterFloodMaps.zip",
        "NYC Stormwater Flood Maps (moderate 2.13 in/hr, current sea level), summarised "
        "to MODZCTA as flooded-area fraction. The Ida / pluvial term.", heavy=True)
def fetch_stormwater():
    import geopandas as gpd

    RAW.mkdir(parents=True, exist_ok=True)
    blob = RAW / "nyc_stormwater_flood_maps.zip"
    url = ("https://data.cityofnewyork.us/api/views/9i7c-xyvv/files/"
           "6ce7b252-a38c-47ae-a823-680f443227e5?filename=NYCFloodStormwaterFloodMaps.zip")
    if not blob.exists() or blob.stat().st_size < 30_000_000:
        with get(url, stream=True) as r, blob.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)

    with zipfile.ZipFile(blob) as z:
        names = z.namelist()
    # Prefer the moderate / current-sea-level layer; fall back to whatever ships.
    cand = [n for n in names if n.lower().endswith((".gdb/", ".shp"))] or names
    target = next((n for n in cand if "moderate" in n.lower() and "2080" not in n), None)
    if target is None:
        print(f"    stormwater: no moderate layer found in {names[:8]}")
        return None

    extract = RAW / "stormwater"
    extract.mkdir(exist_ok=True)
    with zipfile.ZipFile(blob) as z:
        z.extractall(extract)
    src = next(iter(sorted(extract.rglob("*.shp"))), None) or \
        next(iter(sorted(extract.rglob("*.gdb"))), None)
    if src is None:
        print("    stormwater: no readable vector layer after extract")
        return None

    flood = gpd.read_file(src).to_crs(2263)
    flood["geometry"] = flood.geometry.buffer(0)
    zcta = gpd.read_file(REF / "nyc_modzcta.geojson").to_crs(2263)
    zcta["geometry"] = zcta.geometry.buffer(0)
    zcta["zcta_area"] = zcta.geometry.area

    inter = gpd.overlay(zcta[["modzcta", "zcta_area", "geometry"]], flood, how="intersection")
    inter["frac"] = inter.geometry.area / inter["zcta_area"]
    agg = inter.groupby("modzcta")["frac"].sum().clip(upper=1.0).reset_index()
    agg.columns = ["modzcta", "stormwater_flooded_frac"]
    df = pl.DataFrame(agg.to_dict("list"))
    all_mz = pl.read_parquet(REF / "nyc_modzcta.parquet").select("modzcta")
    df = all_mz.join(df, on="modzcta", how="left").fill_null(0.0)
    return write(df, "stormwater_by_modzcta", url)


@source("floodnet_sensors", "https://data.cityofnewyork.us/resource/kb2e-tjy3.json",
        "FloodNet sensor deployments with ZIP, borough and lat/lon. "
        "NOTE: floodnet.nyc gates its own API behind a request form, but the same "
        "data is open on NYC Open Data. No form needed.")
def fetch_floodnet_sensors():
    df = socrata("data.cityofnewyork.us", "kb2e-tjy3")
    keep = [c for c in ["sensor_name", "sensor_id", "date_installed", "date_removed",
                        "tidally_influenced", "street_name", "borough", "zipcode",
                        "nta", "latitude", "longitude",
                        "lowest_point_height_delta_inches"] if c in df.columns]
    df = df.select(keep).with_columns([
        pl.col("latitude").cast(pl.Float64, strict=False),
        pl.col("longitude").cast(pl.Float64, strict=False)])
    return write(df, "floodnet_sensors", "https://data.cityofnewyork.us/d/kb2e-tjy3")


@source("floodnet_events", "https://data.cityofnewyork.us/resource/aq7i-eu5q.json",
        "FloodNet street-flooding events with max depth and duration. Real observed "
        "pluvial floods, usable as a replay scenario instead of a scripted one.")
def fetch_floodnet_events():
    df = socrata("data.cityofnewyork.us", "aq7i-eu5q")
    num = ["max_depth_inches", "onset_time_mins", "drain_time_mins", "duration_mins",
           "duration_above_4_inches_mins", "duration_above_12_inches_mins",
           "duration_above_24_inches_mins"]
    df = df.select([c for c in df.columns if not c.startswith("flood_profile")])
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False)
                          for c in num if c in df.columns])
    return write(df, "floodnet_events", "https://data.cityofnewyork.us/d/aq7i-eu5q")


@source("va_facilities", "https://services2.arcgis.com/VFLAJVozK0rtzQmT/arcgis/rest/services/"
        "Veterans_Health_Administration_Medical_Facilities/FeatureServer",
        "VHA medical facilities with station number, address and lat/lon. "
        "NOTE: api.va.gov/services/va_facilities needs an approved developer.va.gov key; "
        "this mirror is keyless, which is why the build uses it.")
def fetch_va_facilities():
    svc = ("https://services2.arcgis.com/VFLAJVozK0rtzQmT/arcgis/rest/services/"
           "Veterans_Health_Administration_Medical_Facilities/FeatureServer")
    feats = arcgis(svc, 0, where="S_STATE='NY'",
                   out_fields="STA_NO,PAR_STA_NO,STA_NAME,CNAME,S_ADD1,S_CITY,S_STATE,S_ZIP,LAT,LON")
    df = pl.DataFrame([f["attributes"] for f in feats]).rename(
        {"STA_NO": "station_no", "PAR_STA_NO": "parent_station_no", "STA_NAME": "name",
         "CNAME": "county", "S_ADD1": "address", "S_CITY": "city", "S_STATE": "state",
         "S_ZIP": "zip", "LAT": "lat", "LON": "lon"})
    df = df.with_columns([pl.col("lat").cast(pl.Float64, strict=False),
                          pl.col("lon").cast(pl.Float64, strict=False)])
    return write(df, "va_facilities_ny", svc)


@source("va_facility_hazard", "derived: va_facilities_ny x epne-qv9x x stormwater",
        "The SiteDown input table: every NYC VA facility with the hurricane evacuation "
        "zone and stormwater-flood area it sits in. This is the join that shows station "
        "630, the Manhattan VA that evacuated on 28 Oct 2012, is in evacuation zone 1.")
def fetch_va_facility_hazard():
    import geopandas as gpd
    from shapely.geometry import Point, shape

    fac = pl.read_parquet(REF / "va_facilities_ny.parquet").filter(
        pl.col("county").str.to_uppercase().is_in(list(NYC_COUNTY_NAMES)))
    fac = fac.with_columns(
        pl.col("county").str.to_uppercase().replace_strict(NYC_COUNTY_NAMES, default=None)
          .alias("borough"))

    raw = socrata_rows("data.cityofnewyork.us", "epne-qv9x")
    zones = gpd.GeoDataFrame(
        [{"evac_zone": int(r["hurricane_"]), "geometry": shape(r["the_geom"])}
         for r in raw if r.get("the_geom") and str(r.get("hurricane_", "")).isdigit()],
        crs="EPSG:4326")

    pdf = fac.to_pandas()
    g = gpd.GeoDataFrame(pdf, crs="EPSG:4326",
                         geometry=[Point(x, y) for x, y in zip(pdf["lon"], pdf["lat"])])
    j = gpd.sjoin(g, zones[["evac_zone", "geometry"]], how="left", predicate="within")
    j = j.drop(columns=["geometry", "index_right"]).drop_duplicates("station_no")

    df = pl.DataFrame(j.astype({"address": "string"}).to_dict("list"),
                      strict=False).with_columns(
        pl.col("evac_zone").cast(pl.Int8, strict=False).fill_null(0))

    # Attach the ZIP-level stormwater fraction if that fetcher has already run.
    sw = REF / "stormwater_by_modzcta.parquet"
    if sw.exists():
        df = df.join(pl.read_parquet(sw), left_on="zip", right_on="modzcta", how="left")

    # site_dependent_services marks the facilities that hold dialysis / infusion / OTP,
    # i.e. the ones whose closure creates a treatment gap rather than an inconvenience.
    df = df.with_columns(
        pl.col("station_no").is_in(["630", "630A4", "526"]).alias("site_dependent_services"))
    return write(df, "va_facilities_nyc_hazard", "derived join, see docs/sources.md")


@source("cooling_sites", "https://data.cityofnewyork.us/resource/h2bn-gu9k.json",
        "NYC Parks Cool It! cooling sites (misting stations, spray showers, cool spots). "
        "Destination set for the cooling-center-ride action.")
def fetch_cooling_sites():
    df = socrata("data.cityofnewyork.us", "h2bn-gu9k")
    return write(df, "nyc_cooling_sites", "https://data.cityofnewyork.us/d/h2bn-gu9k")


@source("airnow_smoke", "https://files.airnowtech.org/airnow/2023/20230607/daily_data_v2.dat",
        "EPA AirNow daily files for the June 2023 Canadian-wildfire smoke episode plus a "
        "recent baseline week. NOTE: airnowapi.org needs a free key, but "
        "files.airnowtech.org is keyless and carries the same monitor data.")
def fetch_airnow_smoke():
    cols = ["date", "aqsid", "site", "parameter", "units", "value", "averaging_hours",
            "agency", "aqi", "aqi_category", "lat", "lon", "full_aqsid"]
    frames, got = [], []
    days = [f"2023060{d}" for d in range(5, 10)] + ["20230610", "20230611"]
    for day in days:
        url = f"https://files.airnowtech.org/airnow/{day[:4]}/{day}/daily_data_v2.dat"
        try:
            txt = get(url).text
        except Exception as exc:  # noqa: BLE001
            print(f"    airnow {day}: {exc}")
            continue
        df = pl.read_csv(io.StringIO(txt), separator="|", has_header=False,
                         new_columns=cols, infer_schema_length=0)
        frames.append(df)
        got.append(day)
    if not frames:
        return None
    df = pl.concat(frames, how="vertical_relaxed")
    df = df.filter(pl.col("parameter").str.contains("PM2.5"))
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False)
                          for c in ["value", "aqi", "lat", "lon"]])
    # Keep the NYC metro box; the model uses monitor-to-ZIP nearest join.
    df = df.filter((pl.col("lat").is_between(40.4, 41.1)) & (pl.col("lon").is_between(-74.3, -73.6)))
    res = write(df, "airnow_pm25_nyc_smoke2023", "https://files.airnowtech.org/airnow/")
    res["days"] = got
    return res


@source("nws_snapshot", "https://api.weather.gov/alerts/active?area=NY",
        "A live snapshot of NWS active alerts and the 7-day gridpoint forecast for the "
        "five borough points. Keyless. Re-run on demo morning; the snapshot is the "
        "offline fallback.")
def fetch_nws_snapshot():
    points = {"Manhattan": (40.7831, -73.9712), "Bronx": (40.8448, -73.8648),
              "Brooklyn": (40.6782, -73.9442), "Queens": (40.7282, -73.7949),
              "Staten Island": (40.5795, -74.1502)}
    alerts = get("https://api.weather.gov/alerts/active", params={"area": "NY"}).json()
    (RAW / "nws_alerts_ny.json").parent.mkdir(parents=True, exist_ok=True)
    (REF / "nws_alerts_ny.json").write_text(json.dumps(alerts, indent=1))

    rows = []
    for boro, (lat, lon) in points.items():
        meta = get(f"https://api.weather.gov/points/{lat},{lon}").json()["properties"]
        fc = get(meta["forecast"]).json()["properties"]["periods"]
        for p in fc:
            rows.append({"borough": boro, "grid": f"{meta['gridId']}/{meta['gridX']},{meta['gridY']}",
                         "start": p["startTime"], "end": p["endTime"], "is_day": p["isDaytime"],
                         "temp_f": p["temperature"], "short": p["shortForecast"],
                         "detailed": p["detailedForecast"]})
    df = pl.DataFrame(rows)
    res = write(df, "nws_forecast_nyc", "https://api.weather.gov/")
    res["active_alerts"] = len(alerts.get("features", []))
    return res


@source("acs_veterans", "https://www2.census.gov/programs-surveys/acs/summary_file/2023/"
        "table-based-SF/data/5YRData/acsdt5y2023-b21001.dat",
        "ACS 5-year B21001 (sex by age by veteran status) per ZCTA, used to weight where "
        "the synthetic cohort is re-homed. "
        "NOTE: api.census.gov now 302s to missing_key.html for every request, so the "
        "keyed API is NOT a demo-day dependency; this Summary File path is keyless.",
        heavy=True)
def fetch_acs_veterans():
    RAW.mkdir(parents=True, exist_ok=True)
    geo_path, dat_path = RAW / "acs2023_geos.txt", RAW / "acsdt5y2023-b21001.dat"
    for path, url in [
        (geo_path, "https://www2.census.gov/programs-surveys/acs/summary_file/2023/"
                   "table-based-SF/documentation/Geos20235YR.txt"),
        (dat_path, "https://www2.census.gov/programs-surveys/acs/summary_file/2023/"
                   "table-based-SF/data/5YRData/acsdt5y2023-b21001.dat")]:
        if not path.exists():
            with get(url, stream=True) as r, path.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)

    geos = pl.read_csv(geo_path, separator="|", infer_schema_length=0, truncate_ragged_lines=True)
    gcol = next(c for c in geos.columns if c.upper() in {"GEO_ID", "GEOID"})
    ncol = next(c for c in geos.columns if c.upper() in {"NAME", "GEONAME"})
    zc = geos.filter(pl.col(gcol).str.contains("860")).select(
        pl.col(gcol).alias("geo_id"), pl.col(ncol).alias("name"))
    zc = zc.with_columns(pl.col("geo_id").str.extract(r"(\d{5})$").alias("zcta"))

    dat = pl.read_csv(dat_path, separator="|", infer_schema_length=0, truncate_ragged_lines=True)
    dcol = next(c for c in dat.columns if c.upper() in {"GEO_ID", "GEOID"})
    dat = dat.join(zc, left_on=dcol, right_on="geo_id", how="inner")

    zctas = _nyc_zctas()
    dat = dat.filter(pl.col("zcta").is_in(list(zctas)))
    # B21001 is SEX BY AGE BY VETERAN STATUS, 39 cells. Verified by identity on the
    # real file: E002 == E005 + E023, and each sex total is the sum of its five
    # veteran age cells. Male veteran cells are 008/011/014/017/020 and female are
    # 026/029/032/035/038, for age bands 18-34, 35-54, 55-64, 65-74, 75+.
    BANDS = {"18_34": (8, 26), "35_54": (11, 29), "55_64": (14, 32),
             "65_74": (17, 35), "75plus": (20, 38)}
    want = {"B21001_E001": "pop_18plus", "B21001_E002": "veterans_total"}
    for band, (m, f) in BANDS.items():
        want[f"B21001_E{m:03d}"] = f"vet_m_{band}"
        want[f"B21001_E{f:03d}"] = f"vet_f_{band}"
    have = {c: n for c, n in want.items() if c in dat.columns}
    df = dat.select([pl.col("zcta")] + [pl.col(c).cast(pl.Float64, strict=False).alias(n)
                                        for c, n in have.items()])
    # The re-homing weight the cohort actually needs: veterans per ZCTA per age band.
    for band in BANDS:
        if {f"vet_m_{band}", f"vet_f_{band}"} <= set(df.columns):
            df = df.with_columns((pl.col(f"vet_m_{band}") + pl.col(f"vet_f_{band}"))
                                 .alias(f"vet_{band}"))
    if {"vet_65_74", "vet_75plus"} <= set(df.columns):
        df = df.with_columns((pl.col("vet_65_74") + pl.col("vet_75plus")).alias("veterans_65plus"))
    return write(df, "acs_veterans_by_zcta", "https://www2.census.gov/programs-surveys/acs/"
                 "summary_file/2023/table-based-SF/data/5YRData/acsdt5y2023-b21001.dat")


@source("synthea_sample", "https://synthetichealth.github.io/synthea-sample-data/downloads/"
        "latest/synthea_sample_data_fhir_latest.zip",
        "Synthea FHIR R4 sample bundles (~30 MB). The fixture the FHIR reader and its "
        "tests run against, so Track A is never blocked on the 4 GB VA release.")
def fetch_synthea_sample():
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / "synthea_sample_fhir.zip"
    url = ("https://synthetichealth.github.io/synthea-sample-data/downloads/latest/"
           "synthea_sample_data_fhir_latest.zip")
    if not out.exists():
        with get(url, stream=True) as r, out.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    with zipfile.ZipFile(out) as z:
        n = len([x for x in z.namelist() if x.endswith(".json")])
    return {"file": out.name, "url": url, "rows": n, "cols": 0,
            "sha256": "", "bytes": out.stat().st_size, "note": f"{n} FHIR bundles",
            "fetched_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}


def _nyc_zctas() -> set[str]:
    """Every ZCTA that makes up an NYC MODZCTA, plus the MODZCTA codes themselves."""
    p = REF / "nyc_modzcta.parquet"
    if not p.exists():
        raise SystemExit("run the 'modzcta' fetcher first: it is the geometry base")
    df = pl.read_parquet(p)
    out = set(df["modzcta"].to_list())
    for members in df["zcta_members"].to_list():
        if members:
            out.update(m.strip() for m in str(members).split(","))
    return {z for z in out if z and z.isdigit()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run just these fetchers")
    ap.add_argument("--heavy", action="store_true", help="include multi-hundred-MB sources")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for name, f in REGISTRY.items():
            print(f"{'HEAVY ' if f.heavy else '      '}{name:20s} {f.note.splitlines()[0]}")
        return 0

    REF.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    # modzcta is the geometry base everything else joins to, so it runs first.
    order = ["modzcta"] + [n for n in REGISTRY if n != "modzcta"]
    names = args.only or order
    for name in names:
        f = REGISTRY.get(name)
        if f is None:
            print(f"!! unknown fetcher {name}")
            continue
        if f.heavy and not args.heavy and not args.only:
            print(f"-- {name}: skipped (heavy; pass --heavy)")
            continue
        print(f">> {name}")
        try:
            res = f.fn()
        except Exception:  # noqa: BLE001
            print(f"!! {name} failed, keeping any existing snapshot")
            traceback.print_exc(limit=3)
            continue
        if res is None:
            print(f"-- {name}: no output")
            continue
        res["note"] = f.note
        manifest[name] = res
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"   {res['file']}  rows={res['rows']}  {res['bytes'] / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

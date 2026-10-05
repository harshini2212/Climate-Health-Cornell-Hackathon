"""NYC Environment & Health Data Portal: heat ED visits and hospitalizations by neighbourhood.

The portal's data explorer is a JavaScript page; the numbers behind it are published as JSON
in NYC Health's own repo, which is what this reads (Apache-2.0):

    https://github.com/nychealth/EHDP-data   indicators/data/<IndicatorID>.json

Four indicators, every geography and period the portal has:

* 2443 *Heat stress: 5-year emergency department visits*  2018-22, by CD and borough
* 2410 *Heat stress: 10-year hospitalizations*            2013-22, by CD and borough
* 2075 *Heat stress: yearly emergency department visits*  2005-2022; UHF42 only to 2014
* 2076 *Heat stress: yearly hospitalizations*             2000-2022; UHF42 only to 2016

So the only neighbourhood-level heat outcome that overlaps the back-test's 2017-2021 is
2443, by community district. None of these geographies is `modzcta`, so this module also
writes `ehdp_geo_by_modzcta`: each MODZCTA's UHF42 (from the portal's own `zcta_to_uhf.csv`)
and community district (largest area overlap with the portal's `CD.geojson`).

Counts under 11 are suppressed by SPARCS. They stay null with `suppressed = True`: a
suppressed neighbourhood is "between 1 and 10", never zero.

Note: the explorer link `.../weather-related-illness/?id=2445` is indicator 2445, *Cold
stress: 5-year hospitalizations*. Its heat counterparts are 2443 and 2410.
"""

from __future__ import annotations

import io
from collections import Counter
from collections.abc import Iterable, Mapping

import polars as pl
import requests

REPO = "https://github.com/nychealth/EHDP-data"
#: Pinned: `production` moves most days. This is its head as read on 2026-10-04 (committed
#: 2026-10-02). Bump it deliberately, re-fetch, and re-check the manifest hashes.
COMMIT = "08d6e68f6e1d744f31b54401ec3165520681bb95"
RAW = f"https://raw.githubusercontent.com/nychealth/EHDP-data/{COMMIT}/"
EXPLORER = ("https://a816-dohbesp.nyc.gov/IndicatorPublic/data-explorer/"
            "weather-related-illness/")
SOURCE = "nychealth/EHDP-data"
INDICATORS = {2443: "heat_ed_visits", 2410: "heat_hospitalizations",
              2075: "heat_ed_visits", 2076: "heat_hospitalizations"}
STEM = "ehdp_heat_by_geo"
XWALK_STEM = "ehdp_geo_by_modzcta"


def _get(path: str):
    r = requests.get(RAW + path, timeout=120)
    r.raise_for_status()
    return r


def _find_indicators(metadata) -> dict[int, dict]:
    """The metadata file nests indicators under topics; collect every one by id."""
    found: dict[int, dict] = {}

    def walk(x):
        if isinstance(x, dict):
            if "IndicatorID" in x and "Measures" in x:
                found[int(x["IndicatorID"])] = x
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(metadata)
    return found


def parse(payloads: Mapping[int, dict], metadata, time_periods: Iterable[dict],
          geo_lookup: pl.DataFrame) -> pl.DataFrame:
    """{IndicatorID: columnar JSON} -> one row per indicator x measure x geography x period."""
    meta = _find_indicators(metadata)
    periods = {int(t["TimePeriodID"]): str(t["TimePeriod"]) for t in time_periods}
    names = geo_lookup.select(geo_type=pl.col("GeoType"), geo_id=pl.col("GeoID").cast(pl.String),
                              geo_name=pl.col("Name"))
    frames = []
    for ind, payload in payloads.items():
        if ind not in meta:
            raise ValueError(f"indicator {ind} is not in the EHDP metadata")
        measures = {int(m["MeasureID"]): m["MeasurementType"] for m in meta[ind]["Measures"]}
        df = pl.DataFrame({k: payload[k] for k in
                           ("MeasureID", "GeoID", "GeoType", "TimePeriodID", "Value", "Note")},
                          schema_overrides={"Value": pl.Float64, "Note": pl.String})
        unknown = set(df["MeasureID"].unique()) - set(measures)
        if unknown:
            raise ValueError(f"indicator {ind}: measures {sorted(unknown)} not in metadata")
        frames.append(df.select(
            indicator_id=pl.lit(ind, dtype=pl.Int64),
            outcome=pl.lit(INDICATORS.get(ind, meta[ind]["IndicatorName"])),
            measure_id=pl.col("MeasureID").cast(pl.Int64),
            measure=pl.col("MeasureID").replace_strict(measures, return_dtype=pl.String),
            geo_type=pl.col("GeoType"),
            geo_id=pl.col("GeoID").cast(pl.String),
            time_period=pl.col("TimePeriodID").replace_strict(periods, return_dtype=pl.String),
            value=pl.col("Value"),
            suppressed=pl.col("Value").is_null()
                       & pl.col("Note").fill_null("").str.contains("suppressed"),
        ))
    out = (pl.concat(frames)
           .with_columns(year=pl.when(pl.col("time_period").str.contains(r"^\d{4}$"))
                         .then(pl.col("time_period").cast(pl.Int64, strict=False)))
           .join(names, on=["geo_type", "geo_id"], how="left", maintain_order="left")
           .with_columns(source=pl.lit(SOURCE)))
    return out.select("indicator_id", "outcome", "measure_id", "measure", "geo_type", "geo_id",
                      "geo_name", "time_period", "year", "value", "suppressed", "source")


def crosswalk(zcta_to_uhf: pl.DataFrame, modzcta: pl.DataFrame,
              cd_overlap: pl.DataFrame) -> pl.DataFrame:
    """modzcta -> (uhf42, cd).

    uhf42: the UHF holding most of the MODZCTA's member ZCTAs (ties: the lowest code).
    cd: the community district with the largest area overlap, from `cd_overlap`
    (modzcta, cd, area) -- see `overlap`. Raises naming any MODZCTA it cannot place.
    """
    uhf = dict(zip(zcta_to_uhf["zcta"].cast(pl.String).to_list(),
                   zcta_to_uhf["uhfcode"].cast(pl.String).to_list(), strict=True))
    cd = (cd_overlap.with_columns(pl.col("cd").cast(pl.String))
          .sort(["modzcta", "area", "cd"], descending=[False, True, False])
          .group_by("modzcta", maintain_order=True).first().select("modzcta", "cd"))
    cd_of = dict(zip(cd["modzcta"].to_list(), cd["cd"].to_list(), strict=True))
    rows, missing = [], []
    for code, members in zip(modzcta["modzcta"], modzcta["zcta_members"], strict=True):
        zctas = {code, *(m.strip() for m in str(members or "").split(",") if m.strip())}
        votes = Counter(uhf[z] for z in zctas if z in uhf)
        if not votes or code not in cd_of:
            missing.append(code)
            continue
        top = max(votes.values())
        rows.append((code, min(u for u, n in votes.items() if n == top), cd_of[code]))
    if missing:
        raise ValueError(f"no UHF42 or community district for modzcta {missing}")
    return pl.DataFrame(rows, schema={"modzcta": pl.String, "uhf42": pl.String,
                                      "cd": pl.String}, orient="row")


def overlap(modzcta_geojson: str, cd_geojson: str) -> pl.DataFrame:
    """(modzcta, cd, area in sq ft) for every MODZCTA x CD pair that intersects.

    Needs the `geo` extra, so it runs at fetch time only; `crosswalk` is the tested part.
    """
    import geopandas as gpd

    mz = gpd.read_file(io.StringIO(modzcta_geojson))[["modzcta", "geometry"]].to_crs(2263)
    cd = gpd.read_file(io.StringIO(cd_geojson))[["GEOCODE", "geometry"]].to_crs(2263)
    both = gpd.overlay(mz, cd, how="intersection", keep_geom_type=True)
    return pl.DataFrame({"modzcta": both["modzcta"].astype(str).tolist(),
                         "cd": both["GEOCODE"].astype(str).tolist(),
                         "area": both.geometry.area.tolist()})


def fetch() -> pl.DataFrame:
    payloads = {ind: _get(f"indicators/data/{ind}.json").json() for ind in INDICATORS}
    metadata = _get("indicators/metadata/metadata.json").json()
    periods = _get("indicators/metadata/TimePeriods.json").json()
    geo = pl.read_csv(io.StringIO(_get("geography/GeoLookup.csv").text), infer_schema=False)
    return parse(payloads, metadata, periods, geo)


def fetch_crosswalk(modzcta: pl.DataFrame, modzcta_geojson: str) -> pl.DataFrame:
    z2u = pl.read_csv(io.StringIO(_get("geography/zcta_to_uhf.csv").text), infer_schema=False)
    return crosswalk(z2u, modzcta, overlap(modzcta_geojson, _get("geography/CD.geojson").text))

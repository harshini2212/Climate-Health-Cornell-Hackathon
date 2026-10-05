"""A region: the place a cohort lives in, read from `regions/<id>.yaml`.

    from leeward.geo import region
    nyc = region.get("nyc")              # regions/nyc.yaml, cached
    nyc.ref("places")                    # a reference table, its unit column renamed geo_id
    nyc.subregions                       # the fairness-floor groups, e.g. the five boroughs
    toy = region.load_file(path)         # any other yaml; registered so get("toy") finds it

Every contract table carries `region_id` and `geo_id`. `geo_id` is the region's own unit
(NYC: MODZCTA); it means nothing without the `region_id` beside it, so two regions may
share a code and never a row.

Nothing here reaches the network. A region is a yaml file and the reference tables it
names, and they are read only when asked for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
REGIONS = ROOT / "regions"

#: The region every file written before regions existed belongs to.
DEFAULT = "nyc"

#: The contract name of the unit column, in every table and every frame the pipeline builds.
GEO_ID = "geo_id"

_ID = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class RegionError(ValueError):
    """A region yaml, or a request for a region, that does not make sense."""


@dataclass(frozen=True)
class RefTable:
    path: Path
    #: The column that holds the unit id, renamed to `geo_id` on read. None: not unit-keyed.
    key: str | None = None


@dataclass(frozen=True)
class GeoRange:
    name: str
    lo: int | None = None
    hi: int | None = None

    def holds(self, code: int) -> bool:
        return ((self.lo is None or code >= self.lo)
                and (self.hi is None or code <= self.hi))


@dataclass(frozen=True, eq=False)
class Region:
    id: str
    label: str
    unit: str
    unit_doc: str
    unit_table: str
    #: What a subregion is called here ("borough"); the cohort column keeps that name.
    subregion_name: str
    subregions: tuple[str, ...]
    subregion_ranges: tuple[GeoRange, ...]
    county_fips: dict[str, str]
    county_names: dict[str, str]
    geojson: Path
    geojson_property: str
    reference_dir: Path
    tables: dict[str, RefTable] = field(repr=False)
    source: Path = field(repr=False)

    def path(self, name: str) -> Path:
        """Where reference table `name` lives on disk."""
        if name not in self.tables:
            raise RegionError(f"region {self.id!r} has no reference table {name!r}; "
                              f"it declares {sorted(self.tables)} in {self.source}")
        return self.tables[name].path

    def ref(self, name: str, columns: list[str] | None = None) -> pl.DataFrame:
        """Reference table `name`, with its unit column (if it has one) renamed `geo_id`."""
        t = self.tables.get(name)
        path = self.path(name)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; region {self.id!r} needs it "
                                    f"(scripts/fetch_sources.py builds the NYC ones)")
        rename = t is not None and t.key is not None and (columns is None or GEO_ID in columns)
        if columns is not None and rename and t is not None and t.key:
            columns = [t.key if c == GEO_ID else c for c in columns]
        df = pl.read_parquet(path, columns=columns)
        if rename and t is not None and t.key:
            df = df.rename({t.key: GEO_ID}).with_columns(pl.col(GEO_ID).cast(pl.Utf8))
        return df

    @cached_property
    def geo_ids(self) -> frozenset[str]:
        """Every unit the region has. A geo_id outside this set is not a place."""
        return frozenset(self.ref(self.unit_table, columns=[GEO_ID])[GEO_ID].to_list())

    def subregion_of(self, geo_id: str) -> str:
        """The subregion a unit belongs to, by the yaml's numeric code ranges."""
        if not self.subregion_ranges or not geo_id.isdigit():
            raise RegionError(f"region {self.id!r} assigns subregions by numeric code range, "
                              f"and cannot place {geo_id!r}: add `by_geo_range` to {self.source}")
        code = int(geo_id)
        for r in self.subregion_ranges:
            if r.holds(code):
                return r.name
        raise RegionError(f"region {self.id!r}: no subregion range holds {geo_id!r}")

    def summary(self) -> dict[str, Any]:
        return {"region_id": self.id, "label": self.label, "unit": self.unit,
                "subregion": self.subregion_name, "subregions": list(self.subregions),
                "geojson_property": self.geojson_property}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, Region] = {}


def _need(raw: dict, key: str, where: Path) -> Any:
    if key not in raw:
        raise RegionError(f"{where}: a region needs '{key}'")
    return raw[key]


def parse(raw: dict, source: Path) -> Region:
    """A Region from an already-read yaml mapping. Paths resolve against `source`'s folder."""
    if not isinstance(raw, dict):
        raise RegionError(f"{source}: expected a mapping")
    rid = str(_need(raw, "id", source))
    if not _ID.match(rid):
        raise RegionError(f"{source}: region id {rid!r} must be lower-case letters, digits "
                          "and underscores, starting with a letter")
    base = source.parent
    ref_dir = (base / str(_need(raw, "reference_dir", source))).resolve()
    unit, subs, geo = (_need(raw, k, source) for k in ("unit", "subregions", "geojson"))
    tables = {name: RefTable(path=ref_dir / str(t["path"]), key=t.get("key"))
              for name, t in (_need(raw, "tables", source) or {}).items()}
    unit_table = str(unit.get("table", "units"))
    if unit_table not in tables:
        raise RegionError(f"{source}: unit table {unit_table!r} is not among its tables")

    values = tuple(str(v) for v in _need(subs, "values", source))
    ranges = tuple(GeoRange(name=str(r["name"]), lo=r.get("lo"), hi=r.get("hi"))
                   for r in subs.get("by_geo_range") or [])
    stray = sorted({r.name for r in ranges} - set(values))
    if stray:
        raise RegionError(f"{source}: subregion ranges name {stray}, not in {list(values)}")
    if len(set(values)) != len(values) or not values:
        raise RegionError(f"{source}: subregion values must be non-empty and distinct")

    return Region(
        id=rid, label=str(raw.get("label", rid)),
        unit=str(_need(unit, "name", source)), unit_doc=str(unit.get("doc", "")),
        unit_table=unit_table,
        subregion_name=str(subs.get("name", "subregion")), subregions=values,
        subregion_ranges=ranges,
        county_fips={str(k): str(v) for k, v in (raw.get("county_fips") or {}).items()},
        county_names={str(k): str(v) for k, v in (raw.get("county_names") or {}).items()},
        geojson=ref_dir / str(_need(geo, "path", source)),
        geojson_property=str(_need(geo, "property", source)),
        reference_dir=ref_dir, tables=tables, source=source,
    )


def load_file(path: Path | str) -> Region:
    """Read a region yaml and register it, so `get(its id)` finds it from then on."""
    path = Path(path).resolve()
    r = parse(yaml.safe_load(path.read_text(encoding="utf-8")), path)
    _REGISTRY[r.id] = r
    return r


def get(region_id: str = DEFAULT) -> Region:
    """A registered region, or `regions/<region_id>.yaml`."""
    if region_id in _REGISTRY:
        return _REGISTRY[region_id]
    if not isinstance(region_id, str) or not _ID.match(region_id):
        raise RegionError(f"{region_id!r} is not a region id")
    path = REGIONS / f"{region_id}.yaml"
    if not path.exists():
        raise RegionError(f"unknown region {region_id!r}: no {path.relative_to(ROOT)} and "
                          f"none registered (known: {available()})")
    return load_file(path)


def available() -> list[str]:
    """Every region `get()` can return: the committed yamls plus any registered."""
    return sorted({p.stem for p in REGIONS.glob("*.yaml")} | set(_REGISTRY))


def forget(region_id: str) -> None:
    """Drop a registered region. For tests that register a throwaway one."""
    _REGISTRY.pop(region_id, None)

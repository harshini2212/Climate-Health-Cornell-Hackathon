"""Cached reads for the API. Nothing here computes anything.

Every route answers from a table `make` already wrote, read through `leeward.schema.read` so
the contract is checked on the way in. A file is read once and served from memory until it
changes on disk (mtime and size), so re-running `make score` under a live server shows up on
the next request rather than after a restart, and no request pays for the 6M-row scores table
twice. A missing file is `DataUnavailable`, which the app turns into a 503 that says what to run.

Not thread-safe by accident: one lock covers every load, so two requests that arrive together
on a cold cache read the file once.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import polars as pl

from leeward import schema
from leeward.decision import severity
from leeward.decision import tau as tau_table

T = TypeVar("T")

#: `report/` sits next to `data/`; `make report` writes report.json into it.
REPORT_DIR = schema.ROOT / "report"

#: How many recent `POST /actions` answers stay lookup-able by `GET /message/{action_id}`.
#: The slider produces action ids that are not in the cached actions table.
ISSUED_KEEP = 16

_LOCK = threading.RLock()
_CACHE: dict[Path, tuple[tuple[int, int], Any]] = {}
_ISSUED: deque[pl.DataFrame] = deque(maxlen=ISSUED_KEEP)


class DataUnavailable(RuntimeError):
    """A table the route needs has not been produced yet."""


class AlreadyLogged(ValueError):
    """The outcome log is append-only and keyed by action_id; this one is already in it."""


def _cached(path: Path, load: Callable[[], T], missing: str) -> T:
    try:
        st = path.stat()
    except FileNotFoundError:
        raise DataUnavailable(missing) from None
    sig = (st.st_mtime_ns, st.st_size)
    with _LOCK:
        hit = _CACHE.get(path)
        if hit is not None and hit[0] == sig:
            return hit[1]
        value = load()
        _CACHE[path] = (sig, value)
        return value


def table(name: str) -> pl.DataFrame:
    """A contract table, validated by `schema.read` the first time and whenever it changes."""
    return _cached(
        schema.TABLES[name].path, lambda: schema.read(name),
        f"data/{name}.parquet is missing. `make fixtures` writes a stand-in; the real "
        "pipeline is `make cohort score`.")


def facilities() -> pl.DataFrame:
    """The 14 NYC VA facilities: station_no, name, lat, lon."""
    path = schema.REFERENCE / "va_facilities_nyc_hazard.parquet"
    return _cached(path, lambda: pl.read_parquet(path, columns=["station_no", "name", "lat", "lon"]),
                   f"{path.name} is missing from data/reference/; it is committed, so `git checkout` it.")


def med_classes() -> pl.DataFrame:
    """VA drug class -> CDC climate mechanism, weight and flags."""
    path = schema.REFERENCE / "med_climate_risk.csv"
    return _cached(path, lambda: pl.read_csv(path),
                   f"{path.name} is missing from data/reference/; it is committed, so `git checkout` it.")


def weights() -> dict[str, float]:
    return _cached(severity.PATH, severity.load, f"{severity.PATH.name} is missing")


def _tau_file() -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Both halves of tau.yaml, parsed once. `_CACHE` is keyed by path, so they share a slot."""
    return _cached(tau_table.PATH, lambda: (tau_table.load(), tau_table.lead_days()),
                   f"{tau_table.PATH.name} is missing")


def tau() -> dict[str, dict[str, float]]:
    return _tau_file()[0]


def lead() -> dict[str, int]:
    """How far ahead of the risk each action has to happen. See `decision/allocate.py`."""
    return _tau_file()[1]


def report() -> dict | None:
    """`report/report.json` as written by `make report`, or None if it has not been run."""
    path = REPORT_DIR / "report.json"
    try:
        return _cached(path, lambda: json.loads(path.read_text()), "")
    except DataUnavailable:
        return None


# --------------------------------------------------------------------------- #
# Actions the slider has produced
# --------------------------------------------------------------------------- #

def remember(actions: pl.DataFrame) -> None:
    """Keep an allocation so its action ids can be resolved by `find_action`."""
    with _LOCK:
        _ISSUED.append(actions)


def find_action(action_id: str) -> dict | None:
    """The action row for an id: from a recent `POST /actions`, else from the cached plan."""
    with _LOCK:
        recent = list(_ISSUED)
    for frame in reversed(recent):
        hit = frame.filter(pl.col("action_id") == action_id)
        if hit.height:
            return hit.row(0, named=True)
    try:
        cached = table("actions")
    except DataUnavailable:
        return None
    hit = cached.filter(pl.col("action_id") == action_id)
    return hit.row(0, named=True) if hit.height else None


# --------------------------------------------------------------------------- #
# The outcome log
# --------------------------------------------------------------------------- #

def append_outcome(row: dict) -> int:
    """Append one row to `outcome_log.parquet` and return how many rows it now has."""
    with _LOCK:
        try:
            log = table("outcome_log")
        except DataUnavailable:
            log = schema.empty("outcome_log")
        if log.filter(pl.col("action_id") == row["action_id"]).height:
            raise AlreadyLogged(f"action {row['action_id']} is already in the outcome log; "
                                "the log is append-only, one row per action")
        schema.write(pl.concat([log, pl.DataFrame([row], schema=log.schema)]), "outcome_log")
        return log.height + 1

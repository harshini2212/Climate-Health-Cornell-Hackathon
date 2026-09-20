"""Where the demo opens. The acceptance test for `leeward/demo.py` and `GET /forecast`.

`sandy_then_heat` runs 120 days and 114 of them are quiet: day 0 is 1 June, nine weeks
before landfall. A board that opens at day 0 opens on an empty ribbon under the headline
"No active alerts in the 7-day window" -- the one screen that makes the product look like
it has nothing to say, and the screen `make demo` used to produce.

These tests hold the opposite. Asked for no particular day, the API answers with a window
that contains the first alert; asked for day 0 it still answers with the calm week, because
"here is a calm week, and here is the same team three days later" is a beat worth keeping.
The dates are the real ones from `scenarios/sandy_then_heat.yaml`, so this fails if the
scenario moves and nobody moves the demo with it.
"""

from __future__ import annotations

import re
import subprocess
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

import tables
from leeward import demo, schema
from leeward.api import main as api_main
from leeward.api import schemas as api
from leeward.api import store

ROOT = Path(__file__).resolve().parents[1]

#: The fixture generator's storm and heat wave, which `tests/tables.py` also builds.
FIRST_ALERT = date(2026, 7, 15)

#: `scenarios/sandy_then_heat.yaml`: day 63 of 120 from 2026-06-01.
LANDFALL = date(2026, 8, 3)

#: The heat wave that arrives while the lights are still out (day 66).
FIRST_HEAT = date(2026, 8, 6)


def _quiet(hazards: pl.DataFrame) -> pl.DataFrame:
    """The same table with every alert switched off. `hot_day` is left alone on purpose."""
    return hazards.with_columns(
        *(pl.lit(False).alias(c) for c in demo.ALERT_FLAGS),
        surge_ft=pl.lit(0.0), evac_zone_ordered=pl.lit(0, dtype=pl.Int32),
        outage_frac=pl.lit(0.0))


# --------------------------------------------------------------------------- #
# leeward/demo.py
# --------------------------------------------------------------------------- #

def test_the_opening_window_contains_the_first_alert() -> None:
    hazards, sites = tables.table("hazards"), tables.table("site_status")
    dates = sorted(hazards["date"].unique().to_list())
    start = demo.opening_day(hazards, sites)
    window = dates[start:start + demo.WINDOW_DAYS]

    assert dates[start] < FIRST_ALERT, "the board must open before the event, not on it"
    assert FIRST_ALERT in window
    assert dates.index(FIRST_ALERT) - start == demo.LEAD_DAYS


def test_the_opening_window_is_never_the_quiet_one() -> None:
    """The regression this file exists for: `day=0` on this table has no alerts at all."""
    hazards, sites = tables.table("hazards"), tables.table("site_status")
    dates = sorted(hazards["date"].unique().to_list())
    start = demo.opening_day(hazards, sites)
    alerts = set(demo.alert_days(hazards, sites))

    assert not alerts & set(dates[0:demo.WINDOW_DAYS]), "day 0 is supposed to be quiet here"
    assert alerts & set(dates[start:start + demo.WINDOW_DAYS])


def test_a_merely_hot_day_is_not_an_alert() -> None:
    """82 F in July is a Tuesday in New York. `hot_day` is true on 114 of the 120 days, so
    a rule that counted it would put the demo straight back on the quiet week."""
    hazards = _quiet(tables.table("hazards"))
    assert hazards["hot_day"].any(), "the fixture is only interesting if it is still hot"
    assert demo.alert_days(hazards) == []
    assert demo.opening_day(hazards) == 0


def test_a_closed_site_is_an_event_even_when_the_sky_is_clear() -> None:
    hazards = _quiet(tables.table("hazards"))
    sites = tables.table("site_status")
    down = sorted(sites.filter(pl.col("site_down"))["date"].unique().to_list())
    dates = sorted(hazards["date"].unique().to_list())

    start = demo.opening_day(hazards, sites)
    assert down[0] in dates[start:start + demo.WINDOW_DAYS]


def test_the_window_is_full_even_when_the_event_is_at_the_end() -> None:
    """An alert on the last day pulls the start back so seven days still fit, rather than
    answering with a two-day window nobody can read a week off."""
    hazards = _quiet(tables.table("hazards"))
    dates = sorted(hazards["date"].unique().to_list())
    late = hazards.with_columns(
        heat_alert=pl.col("date") == pl.lit(dates[-1], dtype=pl.Date))

    start = demo.opening_day(late)
    assert len(dates[start:start + demo.WINDOW_DAYS]) == demo.WINDOW_DAYS
    assert dates[-1] in dates[start:start + demo.WINDOW_DAYS]


def test_the_real_scenario_opens_on_the_week_of_landfall() -> None:
    """The one that matters on stage: `scenarios/sandy_then_heat.yaml`, not a fixture."""
    from leeward.ingest.hazards import assemble, load_scenario

    hazards, sites = assemble(load_scenario(ROOT / "scenarios" / "sandy_then_heat.yaml"))
    dates = sorted(hazards["date"].unique().to_list())
    assert dates[0] == date(2026, 6, 1) and len(dates) == 120

    start = demo.opening_day(hazards, sites)
    window = dates[start:start + demo.WINDOW_DAYS]

    assert dates[start] == date(2026, 8, 1), "two days of lead in front of landfall"
    assert LANDFALL in window and FIRST_HEAT in window
    assert start == 61, "scenario-relative, so `make demo DAY=61` reproduces it by hand"


# --------------------------------------------------------------------------- #
# GET /forecast
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """The API over the suite's own frames, written once to a directory of their own."""
    mp = pytest.MonkeyPatch()
    data = tmp_path_factory.mktemp("leeward-demo") / "data"
    mp.setattr(schema, "DATA", data)
    mp.setattr(store, "REPORT_DIR", data.parent / "report")
    for name in schema.TABLES:
        schema.write(tables.table(name), name)
    with TestClient(api_main.app) as c:
        yield c
    mp.undo()


def _forecast(client: TestClient, **params) -> api.ForecastResponse:
    r = client.get("/forecast", params=params)
    assert r.status_code == 200, r.text
    return api.ForecastResponse.model_validate(r.json())


def test_forecast_without_a_day_opens_where_something_is_happening(client: TestClient) -> None:
    hazards, sites = tables.table("hazards"), tables.table("site_status")
    fc = _forecast(client)

    assert fc.day == demo.opening_day(hazards, sites, window=api_main.FORECAST_DAYS)
    assert fc.headline and "No active alerts" not in fc.headline
    assert FIRST_ALERT in fc.dates


def test_day_zero_is_still_one_parameter_away(client: TestClient) -> None:
    """The other half of the beat: the same team, the same screen, a calm week."""
    quiet = _forecast(client, day=0)
    assert quiet.day == 0
    assert quiet.dates[0] == tables.START
    assert quiet.headline == "No active alerts in the 7-day window"
    assert not any(f.site_down for f in quiet.facilities)


def test_an_explicit_day_still_wins(client: TestClient) -> None:
    hazards = tables.table("hazards")
    dates = sorted(hazards["date"].unique().to_list())
    fc = _forecast(client, day=9)
    assert fc.day == 9 and fc.dates == dates[9:16]


# --------------------------------------------------------------------------- #
# The demo path. Nothing below runs the UI; these read the sources that decide the
# opening screen, because a hard-coded 0 in one of them is silent and undoes all of it.
# --------------------------------------------------------------------------- #

UI = ROOT / "ui" / "src"


@pytest.mark.parametrize("screen", ["Week.tsx", "Forecast.tsx", "Map.tsx"])
def test_no_screen_hard_codes_the_quiet_week(screen: str) -> None:
    src = (UI / "screens" / screen).read_text(encoding="utf-8")
    hits = re.findall(r"getForecast\(([^)]*)\)", src)
    assert hits, f"{screen} no longer fetches a forecast; move this guardrail"
    for call in hits:
        assert not re.search(r",\s*0\s*$", call), (
            f"{screen} asks for day 0. Pass the `day` prop so `make demo` opens on the "
            "window leeward/demo.py picks, and 0 stays reachable from the topbar.")


def test_make_demo_can_be_pinned_to_a_day() -> None:
    """`make demo DAY=0` is how the calm week is reached without editing a file."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    body = makefile.split("\ndemo:", 1)[1].split("\n\n", 1)[0]
    dev = makefile.split("\ndemo-dev:", 1)[1].split("\n\n", 1)[0]
    assert "?day=$(DAY)" in body, "make demo must pass DAY through to the built bundle"
    assert "VITE_DEMO_DAY" in dev, "make demo-dev must pass DAY through to the Vite dev server"
    assert re.search(r"^DAY\s*\?=", makefile, re.M), "DAY must be overridable on the command line"


def test_a_built_bundle_takes_the_pinned_day_from_the_url() -> None:
    """`VITE_DEMO_DAY` is read when Vite builds, so it cannot steer the committed ui/dist.
    `make demo DAY=0` therefore prints `/?day=0`, and the UI reads it at runtime."""
    src = (UI / "lib" / "demo.ts").read_text(encoding="utf-8")
    assert re.search(r'URLSearchParams\(window\.location\.search\)\.get\("day"\)', src), (
        "demo.ts no longer reads ?day= from the URL, so `make demo DAY=0` opens the wrong week")
    out = subprocess.run(["make", "-n", "demo", "DAY=0"], cwd=ROOT, capture_output=True,
                         text=True).stdout
    assert "?day=0" in out, f"`make demo DAY=0` does not tell you the URL that pins the day:\n{out}"

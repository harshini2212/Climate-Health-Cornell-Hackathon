"""Which day the demo opens on.

`sandy_then_heat` runs 120 days and 114 of them are quiet. Day 0 is 1 June, nine weeks
before landfall, so a board that opens at day 0 opens on the least interesting week in the
file: an empty ribbon under the headline "No active alerts in the 7-day window". That is a
true answer to the wrong question, and it was the first thing anyone saw.

So the opening window is computed, not typed. It comes out of the hazards table itself,
which means it is still right after `make hazards` moves the dates, right for
`ida_flash_flood` and `smoke_2023`, and right for the fixture tables a clean clone starts
with -- three different calendars, one rule.

**The rule: open `LEAD_DAYS` before the first alert-grade day.** Two days rather than zero
because the product is anticipation, and a board that opens on landfall opens on the
morning the team is already too late; two rather than a week because the window still has
to reach the event. For `sandy_then_heat` that is 1 August: two quiet days, landfall on the
3rd, three days of outage, and the first two days of the heat wave, all on one ribbon.

Alert-grade means a warning a care team would act on. A merely hot day is not one --
`hot_day` is the 82 °F hinge, which this scenario's baseline summer crosses on 3 June and
never comes back under, so counting it would put the demo straight back on the quiet week.

Day 0 stays one parameter away: `GET /forecast?day=0`, or `make demo DAY=0`. "Here is a
calm week, and here is the same team three days later" is the other half of the story.
"""

from __future__ import annotations

from datetime import date as Date

import polars as pl

#: Days of lead time in front of the first alert. See the module docstring for why 2.
LEAD_DAYS = 2

#: Days in a forecast window. Matches `leeward.api.main.FORECAST_DAYS`, which passes its
#: own value in; this default is for callers that have no route to ask.
WINDOW_DAYS = 7

#: Boolean hazard columns that are an alert on their own.
ALERT_FLAGS = ("heat_alert", "smoke_alert", "flood_warning", "flash_flood_emergency")

#: Share of a ZIP without power that counts as an event even with no warning out. Con Ed
#: losing a fifth of a ZIP is a call to every veteran on powered equipment in it.
OUTAGE_ALERT_FRAC = 0.2


def _is_alert() -> pl.Expr:
    """One row of `hazards` is alert-grade: a warning, a surge, an ordered evacuation, or
    an outage big enough to matter. Never `hot_day`."""
    flags = pl.any_horizontal(*(pl.col(c) for c in ALERT_FLAGS))
    return (flags
            | (pl.col("surge_ft") > 0)
            | (pl.col("evac_zone_ordered") > 0)
            | (pl.col("outage_frac") >= OUTAGE_ALERT_FRAC))


def alert_days(hazards: pl.DataFrame, site_status: pl.DataFrame | None = None) -> list[Date]:
    """Every date on which any ZIP is under an alert, or any VA site is closed.

    A closure counts even under a clear sky: station 630 being shut is the event for a
    veteran on dialysis whether or not it is still raining.
    """
    days = set(hazards.filter(_is_alert())["date"].to_list())
    if site_status is not None:
        days |= set(site_status.filter(pl.col("site_down"))["date"].to_list())
    return sorted(days)


def opening_day(hazards: pl.DataFrame, site_status: pl.DataFrame | None = None, *,
                lead: int = LEAD_DAYS, window: int = WINDOW_DAYS) -> int:
    """The scenario-relative day index the demo should open on: 0 is the first day in
    `hazards`, which is what `GET /forecast?day=` counts in.

    `LEAD_DAYS` before the first alert, clamped at both ends: never below 0, and never so
    late that fewer than `window` days are left, because a two-column ribbon is not a week.
    A table with no alerts at all opens at day 0 -- there is no better answer, and saying
    so quietly beats inventing one.
    """
    dates = sorted(hazards["date"].unique().to_list())
    alerts = alert_days(hazards, site_status)
    if not dates or not alerts:
        return 0
    first = min(alerts)
    # Counting rather than `.index()`: site_status can carry a date hazards does not.
    start = sum(1 for d in dates if d < first) - lead
    return max(0, min(start, len(dates) - window))


def opening_window(hazards: pl.DataFrame, site_status: pl.DataFrame | None = None, *,
                   lead: int = LEAD_DAYS, window: int = WINDOW_DAYS) -> list[Date]:
    """The dates `opening_day` picks, for callers that want days rather than an index."""
    dates = sorted(hazards["date"].unique().to_list())
    start = opening_day(hazards, site_status, lead=lead, window=window)
    return dates[start:start + window]


def opening_date(hazards: pl.DataFrame, site_status: pl.DataFrame | None = None, *,
                 lead: int = LEAD_DAYS, window: int = WINDOW_DAYS) -> Date:
    """The first day of the opening window: the demo's "today"."""
    return opening_window(hazards, site_status, lead=lead, window=window)[0]


__all__ = ["LEAD_DAYS", "WINDOW_DAYS", "ALERT_FLAGS", "OUTAGE_ALERT_FRAC",
           "alert_days", "opening_day", "opening_window", "opening_date"]

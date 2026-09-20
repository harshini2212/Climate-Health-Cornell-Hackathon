"""Acceptance tests for lead time: an action has a day it must be DONE by (proposal §6).

The risk day and the do-by day are not the same day. If the surge lands Wednesday, the
alternate dialysis site has to be booked Monday; by Wednesday that action cannot avert
anything, so offering it on Wednesday is offering a placebo.

So `actions.date` is the **do-by day** -- the day the care team does the work -- and
`actions.lead_days` says how far ahead of the risk it had to happen. The risk day is
`date + lead_days`. Capacity is consumed on the do-by day, which is what turns the week
board from seven daily lists into a schedule: Monday's forty calls are spent on Monday's
risk, Wednesday's bookings and Saturday's refills all at once.

The helpers come from `test_allocate.py` so both files describe the same five-veteran world.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest
from test_allocate import _cap, _cohort, _scores

from leeward import schema
from leeward.decision import tau as tau_table
from leeward.decision.allocate import allocate, compare, total_eha
from leeward.schema import ACTION_COST_UNIT, DEFAULT_CAPACITY, NEEDS
from tables import table

SURGE = date(2026, 7, 16)
CALM = {k: (0.001, 0.1) for k in NEEDS}

#: "Today", the allocator's answer with every lead set to zero. The comparison the 15%
#: budget below is measured against, and the only honest way to say "versus today" in a
#: test: it is the same code on the same numbers with the schedule switched off.
NO_LEAD = dict.fromkeys(tau_table.leads(), 0)


def _window(risk: dict, days_before: int = 5, day: date = SURGE) -> pl.DataFrame:
    """`days_before` calm days, then one day carrying `risk`. The forecast arrives on the
    first calm day, so every do-by day in between is a day the team could still act on."""
    calm = [_scores({v: CALM for v in risk}, day=day - timedelta(days=i))
            for i in range(days_before, 0, -1)]
    return pl.concat([*calm, _scores(risk, day=day)])


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

def test_every_proposable_action_has_a_lead_time() -> None:
    leads = tau_table.leads()
    assert set(leads) == set(tau_table.load()) | {"check_in_call"}, (
        "every action allocate() can propose needs a lead time, and only those")
    assert all(isinstance(v, int) and v >= 0 for v in leads.values())


def test_the_actions_that_are_useless_same_day_say_so() -> None:
    """The judgement the whole feature rests on, retyped from docs/proposal.md §6."""
    leads = tau_table.leads()
    assert leads["alt_site_booking"] == 2, "the Sandy dialysis lesson: book before the water"
    assert leads["early_refill"] == 5, "a mail-order refill does not arrive the same afternoon"
    assert leads["care_team_call"] == 0 and leads["verified_text"] == 0
    assert leads["check_in_call"] == 0, "a find-out call is worth nothing if it is not today"
    assert leads["evacuation_assist"] == 0, (
        "a flash flood gives no notice; evacuation help must still be offerable on the day")


def test_lead_days_are_editable_yaml_not_code(tmp_path) -> None:
    edited = tmp_path / "tau.yaml"
    edited.write_text(tau_table.PATH.read_text().replace("lead_days: 2}", "lead_days: 4}"))
    assert tau_table.leads(edited)["alt_site_booking"] == 4
    assert tau_table.load(edited) == tau_table.load(), "editing a lead must not move tau"


@pytest.mark.parametrize("bad", [
    "care_team_call: {breathing: .1, heat: 0, mental: 0, treatment_gap: 0, access_loss: 0}",
    "care_team_call: {breathing: .1, heat: 0, mental: 0, treatment_gap: 0, access_loss: 0,"
    " lead_days: -1}",
    "care_team_call: {breathing: .1, heat: 0, mental: 0, treatment_gap: 0, access_loss: 0,"
    " lead_days: 1.5}",
])
def test_a_missing_or_impossible_lead_fails_loudly(tmp_path, bad: str) -> None:
    f = tmp_path / "tau.yaml"
    f.write_text(bad)
    with pytest.raises(ValueError):
        tau_table.leads(f)


# --------------------------------------------------------------------------- #
# The booking lands before the water
# --------------------------------------------------------------------------- #

def test_a_booking_for_a_surge_is_offered_two_days_before_it_and_not_on_the_day() -> None:
    """The prompt's test. One dialysis patient, one surge day, five days of warning."""
    scores = _window({"V1": {"treatment_gap": (0.40, 0.1)}})
    got = allocate(scores, _cohort([{"veteran_id": "V1", "ckd_dialysis": True}]),
                   DEFAULT_CAPACITY)
    schema.validate(got, "actions")

    booking = got.filter(pl.col("action") == "alt_site_booking")
    assert booking.height == 1, "the one dialysis patient gets the one booking"
    assert booking["date"].to_list() == [SURGE - timedelta(days=2)]
    assert booking["lead_days"].to_list() == [2]
    assert got.filter((pl.col("date") == SURGE)
                      & (pl.col("action") == "alt_site_booking")).height == 0

    same_day = got.filter(pl.col("date") == SURGE)["action"].to_list()
    assert "care_team_call" in same_day, "a call on the day still works, so it is still offered"


def test_the_risk_day_is_the_do_by_day_plus_the_lead(fixtures) -> None:
    """The only arithmetic the UI has to do to put two marks on the ribbon."""
    scores, cohort = fixtures
    got = allocate(scores, cohort, DEFAULT_CAPACITY)
    leads = tau_table.leads()
    risk = got.with_columns(
        (pl.col("date") + pl.duration(days=pl.col("lead_days"))).alias("risk_date"))
    assert (risk["lead_days"] == risk["action"].replace_strict(leads)).all()

    scored = set(scores["date"].to_list())
    assert set(risk["risk_date"].to_list()) <= scored, "a risk day nobody scored is a bug"
    assert (risk["risk_date"] >= risk["date"]).all(), "no action is done after the risk lands"


def test_capacity_is_spent_on_the_do_by_day_not_the_risk_day(fixtures) -> None:
    """Monday's calls are spent on Monday's risk and Wednesday's booking at the same time."""
    scores, cohort = fixtures
    got = allocate(scores, cohort, DEFAULT_CAPACITY)
    for (day, bucket), grp in got.group_by("date", "capacity_bucket"):
        assert grp.height <= DEFAULT_CAPACITY[bucket], (
            f"{day}: {grp.height} actions in bucket {bucket}, capacity is "
            f"{DEFAULT_CAPACITY[bucket]}")

    # The refill bucket is where two risk days visibly share one day's capacity: an early
    # refill five days ahead and a switch to local pickup two days ahead, on the same
    # morning, out of the same 200.
    refills = (got.filter(pl.col("capacity_bucket") == "refill")
                  .group_by("date").agg(pl.col("lead_days").unique().sort().alias("leads")))
    assert refills.height > 0
    assert refills["leads"].list.len().max() > 1, (
        "no day mixes lead times inside a bucket, so the schedule is not actually sharing "
        "a day's capacity between risk days")
    # Every work day that still has the whole horizon in front of it does work for several
    # risk days at once. The last days of the window rightly do not: the forecast runs out.
    horizon = max(tau_table.leads().values())
    ahead = got.filter(pl.col("date") <= scores["date"].max() - timedelta(days=horizon))
    assert ahead.group_by("date").agg(pl.col("lead_days").n_unique())["lead_days"].min() > 1


# --------------------------------------------------------------------------- #
# What the schedule costs, and what it admits it cannot do
# --------------------------------------------------------------------------- #

def test_scheduling_costs_at_most_fifteen_percent_of_the_harm_averted(fixtures) -> None:
    """Realism is allowed to cost something. It is not allowed to cost the demo."""
    scores, cohort = fixtures
    today = total_eha(allocate(scores, cohort, DEFAULT_CAPACITY, lead=NO_LEAD))
    scheduled = total_eha(allocate(scores, cohort, DEFAULT_CAPACITY))
    assert today > 0
    assert scheduled >= 0.85 * today, (
        f"the schedule averts {scheduled:.1f} against {today:.1f} with every lead at zero, "
        f"a fall of {100 * (1 - scheduled / today):.1f}%")


def test_an_action_whose_do_by_day_has_passed_is_not_offered_and_is_counted() -> None:
    """A forecast that arrives on the day of the surge cannot buy a booking any more."""
    cohort = _cohort([{"veteran_id": "V1", "ckd_dialysis": True, "n_active_meds": 2}])
    late = _scores({"V1": {"treatment_gap": (0.40, 0.1)}}, day=SURGE)
    got, _, too_late = compare(late, cohort, DEFAULT_CAPACITY)

    assert got.height > 0, "same-day actions still work and must still be offered"
    assert got["lead_days"].max() == 0
    assert too_late > 0, "the actions the forecast was too late for are worth counting"
    assert "alt_site_booking" not in got["action"].to_list()

    in_time = _window({"V1": {"treatment_gap": (0.40, 0.1)}})
    _, _, none_late = compare(in_time, cohort, DEFAULT_CAPACITY)
    assert none_late == 0, "five days of warning is enough for every lead time there is"


def test_nothing_is_scheduled_before_the_forecast_arrived(fixtures) -> None:
    scores, cohort = fixtures
    got = allocate(scores, cohort, DEFAULT_CAPACITY)
    assert got["date"].min() >= scores["date"].min()


# --------------------------------------------------------------------------- #
# A single do-by day: what POST /actions asks for
# --------------------------------------------------------------------------- #

def test_one_do_by_day_is_the_same_list_the_whole_window_puts_on_that_day(fixtures) -> None:
    """`date=` picks a work day out of the window, and must not change what lands on it.

    True because do-by days do not share capacity: each gets the whole team for a day, so
    allocating one needs only the risk days that feed it, which is what keeps the slider
    a single request.
    """
    scores, cohort = fixtures
    whole = allocate(scores, cohort, DEFAULT_CAPACITY)
    # Not the first days of the window: those lose the long leads the window cannot reach,
    # and `as_of` for a single day is that day, so the two answers rightly differ there.
    day = sorted(scores["date"].unique().to_list())[8]
    one = allocate(scores, cohort, DEFAULT_CAPACITY, date=day)
    assert one.sort("rank")["action_id"].to_list() == \
        whole.filter(pl.col("date") == day).sort("rank")["action_id"].to_list()


def test_a_single_do_by_day_still_reaches_forward_for_the_slow_actions(fixtures) -> None:
    scores, cohort = fixtures
    day = sorted(scores["date"].unique().to_list())[8]
    one = allocate(scores, cohort, DEFAULT_CAPACITY, date=day)
    assert one["date"].unique().to_list() == [day]
    assert one["lead_days"].max() > 0, (
        "a work day that only ever does same-day work is not a schedule")


@pytest.mark.parametrize("bucket", sorted(set(ACTION_COST_UNIT.values())))
def test_more_of_any_bucket_never_averts_less_under_a_schedule(fixtures, bucket: str) -> None:
    """The slider's promise survives the do-by split, because do-by days stay independent."""
    scores, cohort = fixtures
    totals = [total_eha(allocate(scores, cohort, dict(DEFAULT_CAPACITY, **{bucket: n})))
              for n in (0, 4, 16)]
    assert totals == sorted(totals), f"{bucket}: {totals}"


@pytest.fixture(scope="module")
def fixtures() -> tuple[pl.DataFrame, pl.DataFrame]:
    return table("scores"), table("cohort")


def test_five_veterans_with_a_schedule_spend_four_days_of_capacity() -> None:
    """The hand-checked five, given warning: the same people, spread over the days their
    actions actually have to happen on, each day getting the same small capacity."""
    risk = {"V1": {"treatment_gap": (0.40, 0.1)}, "V2": {"heat": (0.30, 0.1)}}
    cohort = _cohort([{"veteran_id": "V1", "ckd_dialysis": True}, {"veteran_id": "V2"}])
    got = allocate(_window(risk), cohort, _cap(call=2, booking=1, ride=1, free=10))
    by_day = dict(got.group_by("date").agg(pl.col("action")).iter_rows())
    assert by_day[SURGE - timedelta(days=2)] == ["alt_site_booking"]
    assert by_day[SURGE - timedelta(days=1)] == ["cooling_center_ride"]
    assert sorted(by_day[SURGE]) == ["care_team_call", "care_team_call",
                                     "verified_text", "verified_text"]

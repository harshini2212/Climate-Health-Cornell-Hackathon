"""The Leeward API: eight routes over tables `make` already wrote (SPEC §9).

    uvicorn leeward.api.main:app --port 8000

Every route reads a cached parquet through `leeward.schema.read` (see `store.py`); nothing here
fits, scores or samples, so the `make demo` path needs no network and no key. The one thing a
request computes is `POST /actions`, which is the capacity slider: it calls
`leeward.decision.allocate.compare` on the days of cached scores that one work day can still
act on, and answers in well under 300 ms at 10,000 veterans.

Response bodies are the frozen models in `schemas.py`. The large ones are built as plain dicts
and validated against the model in one pass, then serialized straight to JSON, which is the
same contract as constructing every row by hand and several times faster.

What is honest about the parameters the models carry but the data cannot yet honour:

* `scenario` -- one hazards table is cached and it does not record which scenario built it, so
  the API serves it as `SCENARIO` and refuses any other name rather than mislabel it.
* `prior_scale` -- only the fitted prior (1.0) is cached; 0.5 and 2.0 need posteriors that no
  lane writes yet, so they are refused rather than silently answered with the 1.0 numbers.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date as Date

import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from leeward import demo, schema
from leeward.api import schemas as api
from leeward.api import store
from leeward.decision import tiers
from leeward.decision.allocate import NEED_PHRASE, compare, total_eha
from leeward.outreach.export import partner_sheet
from leeward.schema import DEFAULT_CAPACITY, NEEDS, TIERS

log = logging.getLogger("uvicorn.error")

#: The one scenario the cached hazards table serves. `make hazards` builds it from
#: scenarios/sandy_then_heat.yaml, the demo's main scenario.
SCENARIO = "sandy_then_heat"
PRIOR_SCALE = 1.0
FORECAST_DAYS = 7

#: name -> the cohort column a team working "by the book" sorts on, highest first.
#: `_random` is added per request from a seed keyed by the date, so the same click gives the
#: same number in rehearsal and on stage.
BASELINES = {"rank_by_age": "age", "rank_by_chronic": "n_chronic", "random": "_random"}

CONDITIONS = {
    "copd": "COPD", "asthma": "Asthma", "chf": "Heart failure", "diabetes": "Diabetes",
    "ckd_dialysis": "Kidney disease on dialysis", "active_cancer_tx": "Active cancer treatment",
    "ptsd": "PTSD", "depression": "Depression",
    "pact_presumptive": "PACT Act presumptive condition",
}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Read every table once at boot, so the first click is as fast as the hundredth."""
    for name in ("cohort", "scores", "hazards", "site_status", "actions"):
        try:
            store.table(name)
        except store.DataUnavailable as e:
            log.warning("not loaded at startup: %s", e)
    store.weights(), store.tau(), store.facilities(), store.med_classes()
    yield


app = FastAPI(
    title="Leeward", version="0.1.0", lifespan=lifespan,
    description="Veteran care continuity under climate events. Synthetic people, real places. "
                "Every route reads cached tables; nothing runs inference in a request.")

# The UI proxies /api in dev; this covers a build pointed straight at :8000 (VITE_API_URL).
app.add_middleware(CORSMiddleware, allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
                   allow_methods=["GET", "POST"], allow_headers=["content-type"])


@app.exception_handler(store.DataUnavailable)
async def _data_unavailable(_: Request, exc: store.DataUnavailable) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=503)


@app.exception_handler(schema.SchemaError)
async def _contract_broken(_: Request, exc: schema.SchemaError) -> JSONResponse:
    return JSONResponse({"detail": f"a cached table no longer matches its contract: {exc}"},
                        status_code=500)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _json(model: BaseModel) -> Response:
    return Response(model.model_dump_json(), media_type="application/json")


def _check_scenario(scenario: str) -> None:
    if scenario != SCENARIO:
        raise HTTPException(422, f"scenario {scenario!r} is not cached; the cached scenario "
                                 f"is {SCENARIO!r}")


def _rung(scores: pl.DataFrame) -> int:
    return int(scores["model_rung"][0])


def _dates(frame: pl.DataFrame) -> list[Date]:
    return sorted(frame["date"].unique().to_list())


def _need_scored_day(scores: pl.DataFrame, day: Date) -> pl.DataFrame:
    """One day of scores, or a 404 that says which days there are."""
    today = scores.filter(pl.col("date") == day)
    if today.height == 0:
        days = _dates(scores)
        raise HTTPException(404, f"no scores for {day}; scored days are {days[0]} to {days[-1]}")
    return today


# --------------------------------------------------------------------------- #
# GET /forecast?scenario=&day=
# --------------------------------------------------------------------------- #

@app.get("/forecast", response_model=api.ForecastResponse)
def forecast(scenario: str = SCENARIO, day: int | None = Query(None, ge=0)) -> Response:
    """Hazards for the seven days from scenario day `day` (0 = the scenario's first day).

    Leave `day` out and the route answers with the window the demo opens on, which
    `leeward.demo.opening_day` reads off the cached hazards table: two days in front of the
    first alert. Day 0 of `sandy_then_heat` is 1 June and the storm is on 3 August, so a
    caller that asks for nothing used to get nine weeks of nothing. `day=0` still means the
    scenario's first day, and the response says which day it answered with either way.
    """
    _check_scenario(scenario)
    hazards = store.table("hazards")
    dates = _dates(hazards)
    if day is None:
        day = demo.opening_day(hazards, store.table("site_status"), window=FORECAST_DAYS)
    if day >= len(dates):
        raise HTTPException(404, f"day {day} is past the end of the scenario, which has "
                                 f"{len(dates)} days (0 to {len(dates) - 1})")
    window = dates[day:day + FORECAST_DAYS]
    hz = hazards.filter(pl.col("date").is_in(window)).sort("date", "modzcta")

    # FacilityStatus carries no date, so a site counts as down if it is down on any day of the
    # window: "will this site be there when the veteran needs it?"
    site = (store.table("site_status").filter(pl.col("date").is_in(window))
                 .group_by("facility_id")
                 .agg(pl.col("site_down").any(), pl.col("evac_zone").first(),
                      pl.col("site_dependent_services").first())
                 .join(store.facilities(), left_on="facility_id", right_on="station_no",
                       how="left")
                 .sort("facility_id"))
    facilities = site.select(list(api.FacilityStatus.model_fields)).to_dicts()

    alerts = hz.group_by("date").agg(pl.col("flood_warning").any(), pl.col("heat_alert").any(),
                                     pl.col("smoke_alert").any()).sort("date")
    bits = []
    for r in alerts.to_dicts():
        d = r["date"].strftime("%a %d %b")
        if r["flood_warning"]:
            bits.append(f"coastal flood warning {d}")
        if r["heat_alert"]:
            bits.append(f"heat alert {d}")
        if r["smoke_alert"]:
            bits.append(f"smoke {d}")
    headline = "; ".join(dict.fromkeys(bits)) or "No active alerts in the 7-day window"
    down = [f["name"] for f in facilities if f["site_down"]]
    if down:
        headline += f". Site down: {', '.join(down)}"

    return _json(api.ForecastResponse.model_validate({
        "scenario": scenario, "day": day, "dates": window, "headline": headline,
        "zips": hz.select(list(api.ZipHazard.model_fields)).to_dicts(),
        "facilities": facilities,
    }))


# --------------------------------------------------------------------------- #
# GET /scores?date=&need=
# --------------------------------------------------------------------------- #

@app.get("/scores", response_model=api.ScoresResponse)
def scores(date: Date, need: str) -> Response:
    """Expected count of `need` per ZIP and per facility, with an 80% band.

    The band is the sum of each veteran's 10th and 90th percentile. That is the exact band
    if every veteran's risk moves together, as it does when the uncertainty is in shared
    coefficients; if they moved independently the true band would be narrower.
    """
    if need not in NEEDS:
        raise HTTPException(422, f"need {need!r} is not one of {NEEDS}")
    all_scores = store.table("scores")
    today = _need_scored_day(all_scores, date).filter(pl.col("need") == need)
    joined = today.join(store.table("cohort").select("veteran_id", "modzcta", "facility_id"),
                        on="veteran_id")

    def by(key: str) -> list[dict]:
        agg = (joined.group_by(key)
                     .agg(pl.col("p_mean").sum().round(3).alias("expected_count"),
                          pl.col("p_lo80").sum().round(3).alias("lo80"),
                          pl.col("p_hi80").sum().round(3).alias("hi80"),
                          pl.len().alias("n_panel"))
                     .sort(key))
        return [{"modzcta": r.pop(key), "need": need, **r} for r in agg.to_dicts()]

    return _json(api.ScoresResponse.model_validate({
        "date": date, "need": need, "zips": by("modzcta"), "facilities": by("facility_id"),
        "model_rung": _rung(all_scores),
    }))


# --------------------------------------------------------------------------- #
# GET /veteran/{id}?date=
# --------------------------------------------------------------------------- #

def _why_tier(tier: str, needs: list[api.NeedScore], weights: dict[str, float]) -> str:
    """The tier rule in `tiers.py`, said in one sentence about this veteran's own numbers."""
    top = max(needs, key=lambda n: n.p_mean)
    if tier == "act_now":
        n = max((n for n in needs if weights[n.need] >= tiers.ACT_NOW_MIN_WEIGHT
                 and n.p_mean >= tiers.ACT_NOW_P and n.p_epistemic_share < tiers.EPISTEMIC_CUT),
                key=lambda n: n.p_mean)
        return (f"{NEED_PHRASE[n.need].capitalize()}: {n.p_mean:.0%} chance (80% interval "
                f"{n.p_lo80:.0%}-{n.p_hi80:.0%}), above the {tiers.ACT_NOW_P:.0%} Act-now line "
                f"for a need this severe, and only {n.p_epistemic_share:.0%} of the uncertainty "
                f"is what we do not know about this veteran.")
    if tier == "find_out":
        n = max((n for n in needs if n.p_epistemic_share >= tiers.EPISTEMIC_CUT
                 and n.p_mean >= tiers.FIND_OUT_P), key=lambda n: n.p_mean)
        return (f"{NEED_PHRASE[n.need].capitalize()}: {n.p_mean:.0%} chance, but "
                f"{n.p_epistemic_share:.0%} of the uncertainty is what we do not know about "
                f"this veteran, so a check-in call is worth more than a guess.")
    if tier == "self_serve":
        return (f"Highest single-need risk is {top.p_mean:.0%} ({NEED_PHRASE[top.need]}): above "
                f"the {tiers.SELF_SERVE_P:.0%} line, short of Act-now and Find-out.")
    return (f"Highest single-need risk is {top.p_mean:.0%} ({NEED_PHRASE[top.need]}), under the "
            f"{tiers.SELF_SERVE_P:.0%} line.")


def _medication_notes(v: dict, hazard: dict | None, site_down: bool) -> list[str]:
    """Plain phrases tying the prescription list to what is happening in this ZIP today."""
    notes = []
    hot = hazard is not None and hazard["hot_day"]
    on_day = f" on a {hazard['heat_index_max_f']:.0f}°F heat-index day" if hot else ""
    if v["med_combo_raas_diuretic"]:
        notes.append("ACE inhibitor or ARB plus a diuretic, the combination CDC names as "
                     f"additive heat risk{on_day}")
    if v["med_renal_triple"]:
        notes.append("NSAID on top of a diuretic and an ACE inhibitor or ARB: kidney injury "
                     "risk when dehydrated")
    if v["med_cold_chain"] and hazard is not None and hazard["outage_frac"] > 0:
        notes.append(f"Refrigerated medication while {hazard['outage_frac']:.0%} of this ZIP "
                     "is without power")
    if v["mail_order_pharmacy"] and hazard is not None and hazard["mail_delivery_disrupted"]:
        notes.append(f"Mail-order pharmacy, {v['days_supply_remaining']} days of supply left, "
                     "and mail delivery to this ZIP is disrupted")
    if v["med_controlled"] and site_down:
        notes.append("Controlled substance and the VA station is closed: the retail emergency "
                     "refill does not cover it")
    if v["acb_score"] >= 3:
        notes.append(f"Anticholinergic burden {v['acb_score']}; 3 or more is clinically "
                     "meaningful")
    heat = (store.med_classes()
                 .filter((pl.col("hazard") == "heat")
                         & pl.col("va_class_id").is_in(v["va_drug_classes"]))
                 .sort("weight", "va_class_id", descending=[True, False]).head(2))
    notes += [f"{r['va_class_name']} ({r['va_class_id']}): {r['mechanism'].replace('_', ' ')}"
              for r in heat.to_dicts()]
    return notes


@app.get("/veteran/{veteran_id}", response_model=api.VeteranCard)
def veteran(veteran_id: str, date: Date) -> Response:
    """The card: risk per need with its band and drivers, tier, medications, the plan."""
    vets = store.table("cohort").filter(pl.col("veteran_id") == veteran_id)
    if vets.height == 0:
        raise HTTPException(404, f"no veteran {veteran_id!r} in the cohort")
    v = vets.row(0, named=True)
    today = _need_scored_day(store.table("scores"), date)
    mine = today.filter(pl.col("veteran_id") == veteran_id)
    if mine.height == 0:
        raise HTTPException(404, f"veteran {veteran_id!r} has no scores on {date}")

    by_need = {r["need"]: r for r in mine.to_dicts()}
    needs = []
    for k in NEEDS:
        r = by_need[k]
        drivers = [(r[f"driver_{i}"], r[f"driver_{i}_contrib"]) for i in (1, 2, 3)
                   if r[f"driver_{i}"] is not None and r[f"driver_{i}_contrib"] is not None]
        needs.append(api.NeedScore(
            need=k, p_mean=r["p_mean"], p_lo80=r["p_lo80"], p_hi80=r["p_hi80"],
            p_epistemic_share=r["p_epistemic_share"],
            drivers=[d for d, _ in drivers], driver_contribs=[c for _, c in drivers]))
    weights = store.weights()
    tier = tiers.assign(mine, weights)["tier"][0]

    hazard = (store.table("hazards")
                   .filter((pl.col("date") == date) & (pl.col("modzcta") == v["modzcta"]))
                   .to_dicts() or [None])[0]
    site = store.table("site_status").filter((pl.col("date") == date)
                                             & (pl.col("facility_id") == v["facility_id"]))
    site_down = bool(site["site_down"].any())
    fac = store.facilities().filter(pl.col("station_no") == v["facility_id"])

    try:
        plan = (store.table("actions")
                     .filter((pl.col("date") == date) & (pl.col("veteran_id") == veteran_id))
                     .sort("rank")["rationale"].to_list())
    except store.DataUnavailable:
        plan = []

    return _json(api.VeteranCard(
        veteran_id=veteran_id, name_display=v["name_display"], age=v["age"],
        modzcta=v["modzcta"], borough=v["borough"], facility_id=v["facility_id"],
        facility_name=fac["name"][0] if fac.height else v["facility_id"], date=date,
        tier=tier, why_this_tier=_why_tier(tier, needs, weights), needs=needs,
        medications=api.MedicationFlags(
            n_active_meds=v["n_active_meds"], thermoreg_score=v["med_thermoreg_score"],
            acb_score=v["acb_score"], combo_raas_diuretic=v["med_combo_raas_diuretic"],
            renal_triple=v["med_renal_triple"], cold_chain=v["med_cold_chain"],
            controlled=v["med_controlled"], narrow_ti=v["med_narrow_ti"],
            mail_order_pharmacy=v["mail_order_pharmacy"],
            days_supply_remaining=v["days_supply_remaining"],
            notes=_medication_notes(v, hazard, site_down)),
        conditions=[label for col, label in CONDITIONS.items() if v[col]],
        powered_equipment=v["powered_equipment"], caregiver=v["caregiver"], floor=v["floor"],
        evac_zone=v["evac_zone"], planned_actions=plan, is_synthetic=True))


# --------------------------------------------------------------------------- #
# POST /actions -- the capacity slider
# --------------------------------------------------------------------------- #

@app.post("/actions", response_model=api.ActionsResponse)
def actions(req: api.ActionsRequest) -> Response:
    """The work list for `date` cut at `capacity`, its total EHA, and what the same team
    would have averted working the same candidates oldest-first, most-chronic-first, or in a
    seeded shuffle. `capacity` may be partial: buckets it omits keep their defaults.

    `date` is the do-by day, not the risk day. An alternate-site booking for Wednesday's
    surge is Monday's work, so this reads the days `date` can still act on -- `date` through
    `date + the longest lead in tau.yaml` -- and returns what has to be done on `date`.
    Each do-by day gets the whole team for a day and they share nothing, so one day is still
    one request.
    """
    _check_scenario(req.scenario)
    if req.prior_scale != PRIOR_SCALE:
        raise HTTPException(422, f"prior_scale {req.prior_scale} is not cached; the cached "
                                 f"posterior uses {PRIOR_SCALE}")
    all_scores = store.table("scores")
    _need_scored_day(all_scores, req.date)       # 404 rather than an empty list for a typo
    cohort = store.table("cohort")
    capacity = {**DEFAULT_CAPACITY, **req.capacity}

    shuffle = np.random.default_rng([0, req.date.toordinal()]).random(cohort.height)
    try:
        chosen, baseline, n_too_late = compare(
            all_scores, cohort.with_columns(pl.Series("_random", shuffle)), capacity,
            req.group_floor, rank_by=BASELINES, date=req.date, weights=store.weights(),
            tau=store.tau())
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    store.remember(chosen)

    rows = (chosen.join(cohort.select("veteran_id", "name_display", "modzcta", "borough"),
                        on="veteran_id", how="left")
                  .sort("rank").select(list(api.ActionRow.model_fields)).to_dicts())
    by_tier = dict.fromkeys(TIERS, 0) | dict(chosen.group_by("tier").len().iter_rows())
    total = total_eha(chosen)
    return _json(api.ActionsResponse.model_validate({
        "date": req.date, "capacity": capacity, "actions": rows, "total_eha": total,
        "baselines": [{"name": "leeward", "total_eha": total},
                      *({"name": n, "total_eha": t} for n, t in baseline.items())],
        "n_panel": cohort.height, "n_selected": chosen.height, "counts_by_tier": by_tier,
        "n_too_late": n_too_late, "model_rung": _rung(all_scores),
    }))


# --------------------------------------------------------------------------- #
# GET /message/{action_id}
# --------------------------------------------------------------------------- #

@app.get("/message/{action_id}", response_model=api.Message)
def message(action_id: str) -> api.Message:
    """The outreach text for an action, from `leeward.outreach.messages.render`.

    That module is built in another lane and imported here on demand, so this route answers
    503 until it lands and nothing else in the API waits for it. Either id form works: the
    `action_id` or the `message_id` (`msg-<action_id>`) that `POST /actions` also returns.
    """
    try:
        render = importlib.import_module("leeward.outreach.messages").render
    except (ImportError, AttributeError) as e:
        raise HTTPException(503, "outreach messages are not available yet: "
                                 f"leeward.outreach.messages.render did not import ({e})") from e
    action = store.find_action(action_id.removeprefix("msg-"))
    if action is None:
        raise HTTPException(404, f"no action {action_id!r}; ids come from POST /actions")
    vets = store.table("cohort").filter(pl.col("veteran_id") == action["veteran_id"])
    if vets.height == 0:
        raise HTTPException(404, f"action {action_id!r} is for veteran "
                                 f"{action['veteran_id']!r}, who is not in the cohort")
    try:
        return api.Message.model_validate(render(action=action, veteran=vets.row(0, named=True)))
    except ValidationError as e:
        raise HTTPException(500, f"leeward.outreach.messages.render did not return a Message: "
                                 f"{e}") from e


# --------------------------------------------------------------------------- #
# POST /log
# --------------------------------------------------------------------------- #

@app.post("/log", response_model=api.LogResponse)
def log_outcome(req: api.LogRequest) -> api.LogResponse:
    """Append what happened to `outcome_log.parquet`. Append-only: one row per action."""
    if not req.logged_by.strip():
        raise HTTPException(422, "logged_by must say who logged it")
    if store.table("cohort").filter(pl.col("veteran_id") == req.veteran_id).height == 0:
        raise HTTPException(404, f"no veteran {req.veteran_id!r} in the cohort")
    try:
        n = store.append_outcome(req.model_dump())
    except store.AlreadyLogged as e:
        raise HTTPException(409, str(e)) from e
    return api.LogResponse(ok=True, n_rows=n)


# --------------------------------------------------------------------------- #
# GET /report
# --------------------------------------------------------------------------- #

@app.get("/report", response_model=api.ReportResponse)
def report() -> api.ReportResponse:
    """The eval report from `make report`. Before it has been run there is only the rung, and
    `generated_at` is null -- an empty fairness table then means "not audited", not "passed".
    """
    raw = store.report()
    if raw is None:
        return api.ReportResponse(model_rung=_rung(store.table("scores")))
    try:
        return api.ReportResponse.model_validate(raw)
    except ValidationError as e:
        raise HTTPException(500, f"report/report.json does not match ReportResponse: {e}") from e


# --------------------------------------------------------------------------- #
# GET /export?date=
# --------------------------------------------------------------------------- #

@app.get("/export")
def export(date: Date) -> Response:
    """The partner sheet for the cached plan on `date`, as CSV. Rows without consent are
    excluded, never redacted (`leeward.outreach.export`)."""
    _need_scored_day(store.table("scores"), date)
    plan = store.table("actions").filter(pl.col("date") == date)
    sheet = partner_sheet(actions=plan, cohort=store.table("cohort"))
    return Response(sheet.write_csv(), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="partner_sheet_{date}.csv"'})

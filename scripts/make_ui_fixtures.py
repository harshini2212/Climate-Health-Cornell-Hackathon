"""Write the JSON the UI reads until the API exists, shaped exactly like leeward/api/schemas.py.

    python scripts/make_fixtures.py --start 2026-06-01 --days 120
    python -m leeward.ingest.hazards --scenario scenarios/sandy_then_heat.yaml
    python scripts/make_ui_fixtures.py

Outputs, all under ui/public/fixtures/ and committed (they are small):

    forecast.json             ForecastResponse: the 7 days the demo opens on (see
                              leeward/demo.py, or pass --date/--day), from
                              data/hazards.parquet + data/site_status.parquet
    scores.json               { date: { need: ScoresResponse } } for the same 7 days,
                              from data/scores.parquet aggregated per ZIP
    actions_candidates.json   a ranked candidate list from data/scores.parquet; the UI
                              cuts it at whatever capacity the slider asks for, so the
                              slider tells the real story before allocate.py lands
    veterans.json             { veteran_id: VeteranCard } for everyone in that candidate
                              list, so a click on the week board opens a real card offline
    messages.json             { action_id: Message } for every candidate action, each one
                              carrying all five elements that tell it apart from a scam
    report.json               ReportResponse: report/report.json from the last `make
                              report`, which is gitignored, so the model report screen
                              still has recovery, reliability and the fairness audit in a
                              clean clone

Every object is validated through the pydantic models before it is written, so the UI
is typed against the contract and not against whatever this script happened to emit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from leeward import demo, schema
from leeward.api import schemas as api
from leeward.api.main import why_tier
from leeward.decision import severity
from leeward.schema import DEFAULT_CAPACITY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ui" / "public" / "fixtures"

#: Severity weights w_k, the same defaults decision/severity.py will carry.
SEVERITY = {"breathing": 3.0, "heat": 4.0, "mental": 4.0, "treatment_gap": 5.0, "access_loss": 3.0}



def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=None, separators=(",", ":"), default=str) + "\n",
                    encoding="utf-8")
    print(f"  {path.relative_to(ROOT).as_posix():44s} {path.stat().st_size / 1e3:8.1f} kB")


def build_forecast(dates: list, day_index: int) -> api.ForecastResponse:
    hz = schema.read("hazards").filter(pl.col("date").is_in(dates))
    # One row per facility per day, like the route: a closure is a fact about a day, so the
    # ribbon can put it on the day it starts rather than over the whole window.
    site = (schema.read("site_status").filter(pl.col("date").is_in(dates))
                  .sort("facility_id", "date"))
    fac = pl.read_parquet(schema.REFERENCE / "va_facilities_nyc_hazard.parquet")
    site = site.join(fac.select("station_no", "name", "lat", "lon"),
                     left_on="facility_id", right_on="station_no", how="left")
    zip_fields = [f for f in api.ZipHazard.model_fields]
    zips = [api.ZipHazard(**{k: r[k] for k in zip_fields}) for r in hz.select(zip_fields).to_dicts()]
    facilities = [api.FacilityStatus(**{k: r[k] for k in api.FacilityStatus.model_fields})
                  for r in site.to_dicts()]
    #: The first day of the window each site is down, so the banner says when, not just who.
    down: dict[str, date] = {}
    for f in facilities:
        if f.site_down and f.name not in down:
            down[f.name] = f.date
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
    if down:
        headline += ". Site down: " + ", ".join(
            f"{name} from {on.strftime('%a %d %b')}" for name, on in down.items())
    return api.ForecastResponse(scenario="sandy_then_heat", day=day_index, dates=dates,
                                zips=zips, facilities=facilities, headline=headline)


def build_scores(cohort: pl.DataFrame, scores: pl.DataFrame, dates: list) -> dict:
    rung = int(scores["model_rung"][0])
    joined = (scores.filter(pl.col("date").is_in(dates))
                    .join(cohort.select("veteran_id", "modzcta"), on="veteran_id"))
    agg = (joined.group_by("date", "need", "modzcta")
                 .agg(pl.col("p_mean").sum().alias("expected_count"),
                      pl.col("p_lo80").sum().alias("lo80"),
                      pl.col("p_hi80").sum().alias("hi80"),
                      pl.len().alias("n_panel"))
                 .sort("date", "need", "modzcta"))
    out: dict[str, dict[str, dict]] = {}
    for (d, need), grp in agg.group_by("date", "need", maintain_order=True):
        rows = [api.ZipScore(modzcta=r["modzcta"], need=need,
                             expected_count=round(r["expected_count"], 3),
                             lo80=round(r["lo80"], 3), hi80=round(r["hi80"], 3),
                             n_panel=r["n_panel"]) for r in grp.to_dicts()]
        resp = api.ScoresResponse(date=d, need=need, zips=rows, model_rung=rung)
        out.setdefault(str(d), {})[need] = resp.model_dump(mode="json")
    return out


def build_candidates(cohort: pl.DataFrame, scores: pl.DataFrame, day, seed: int) -> dict:
    """The real allocator's list at the slider's maximum (100 calls), so the client can cut
    it at any smaller capacity with the same rules and still match what /actions would say.
    Baselines need age, n_chronic and a seeded random key per row, so those ride along.

    `day` is the do-by day, as it is everywhere else: some of these are for risk that lands
    later, and each row's `lead_days` says how much later."""
    from leeward.decision.allocate import allocate

    r = np.random.default_rng(seed)
    cap = dict(DEFAULT_CAPACITY, call=100)
    acts = allocate(scores, cohort, cap, date=day)
    today = scores.filter(pl.col("date") == day)
    top = (today.sort("p_mean", descending=True)
                .group_by("veteran_id", maintain_order=True)
                .agg(pl.col("driver_1").first().alias("top_driver")))
    joined = (acts.join(cohort.select("veteran_id", "name_display", "modzcta", "borough",
                                      "age", "n_chronic"), on="veteran_id", how="left")
                  .join(top, on="veteran_id", how="left")
                  .sort("rank"))
    # Every costed action is kept; the free bucket (verified texts) is thousands of rows
    # a day, so keep the first 300 of those to hold the file under half a megabyte.
    n_free = 0
    rows = []
    for v in joined.to_dicts():
        if v["capacity_bucket"] == "free":
            n_free += 1
            if n_free > 300:
                continue
        rows.append({
            "action_id": v["action_id"], "rank": int(v["rank"]), "veteran_id": v["veteran_id"],
            "name_display": v["name_display"], "modzcta": v["modzcta"], "borough": v["borough"],
            "action": v["action"], "tier": v["tier"], "eha": round(float(v["eha"]), 4),
            "capacity_bucket": v["capacity_bucket"], "owner": v["owner"],
            "lead_days": int(v["lead_days"]),
            "headline": v["headline"], "rationale": v["rationale"],
            "top_driver": v["top_driver"],
            "message_id": v["message_id"],
            "age": int(v["age"]), "n_chronic": int(v["n_chronic"]), "rand": float(r.random()),
        })
    for row in rows:  # every candidate must be a legal ActionRow
        api.ActionRow(**{k: row[k] for k in api.ActionRow.model_fields})
    return {"date": str(day), "model_rung": int(scores["model_rung"][0]),
            "n_panel": cohort.height, "candidates": rows}



# --------------------------------------------------------------------------- #
# The detail views the week board opens. Everything clinical on these two lives
# behind a click, because the board itself hangs on a wall in a shared space.
# --------------------------------------------------------------------------- #

#: Plain words for the cohort's diagnosis booleans. Shown only after a reveal.
CONDITION_LABELS = {
    "copd": "COPD", "asthma": "Asthma", "chf": "Heart failure", "diabetes": "Diabetes",
    "ckd_dialysis": "Dialysis", "active_cancer_tx": "Cancer treatment",
    "ptsd": "PTSD", "depression": "Depression",
}

#: The verification phrase is four ordinary, unmistakable words a veteran reads back.
#: Concrete nouns only: nothing clinical, nothing that hints at why we are calling.
PHRASE_WORDS = (
    "anchor", "harbor", "lantern", "compass", "ferry", "bridge", "beacon", "tide",
    "maple", "copper", "quarry", "meadow", "cedar", "granite", "willow", "amber",
)

CHANNEL_FOR_OWNER = {
    "pharmacist": "care_team_phone", "care_team": "care_team_phone",
    "partner": "care_team_phone", "automated": "VEText",
}

#: What the message actually asks the veteran to do, one line per action.
MESSAGE_ASK = {
    "care_team_call": "Your VA care team will call you today about the days ahead.",
    "check_in_call": "Your VA care team will call for a three-minute check-in today.",
    "backup_power_plan": "We would like to go over a backup-power plan with you before the outage.",
    "cold_chain_plan": "We would like to go over how to keep refrigerated medicine cold if the power goes out.",
    "early_refill": "We can send your refill early so it reaches you before the weather does.",
    "switch_to_local_pickup": "We can move this month's prescription to a pharmacy you can walk to.",
    "cooling_center_ride": "We can book you a ride to a cooling centre on the hot days.",
    "clean_air_room": "We can help you set up one clean-air room at home before the smoke arrives.",
    "alt_site_booking": "Your usual VA site is closed, so we can book your visit at another one.",
    "evacuation_assist": "Your address is in an evacuation zone and we can help you leave safely.",
    "assign_buddy": "A VA-approved community partner can check on you during the storm.",
    "pharmacist_med_review": "A VA clinical pharmacist will review your prescriptions and call you.",
    "controlled_substance_bridge": "We are arranging your medicine directly through the VA pharmacy.",
    "heap_application": "You may qualify for help with your energy bill and we can start the form.",
    "verified_text": "A VA wellness note, with the cooling-site list for your area.",
}


def _phrase(action_id: str) -> str:
    """Four words, fixed by the action id: the same click says the same words on stage."""
    digest = hashlib.sha256(action_id.encode()).digest()
    n = len(PHRASE_WORDS)
    picked: list[str] = []
    for b in digest:
        w = PHRASE_WORDS[b % n]
        if w not in picked:
            picked.append(w)
        if len(picked) == 4:
            break
    return " ".join(picked)


def build_veterans(cohort: pl.DataFrame, scores: pl.DataFrame, candidates: list[dict],
                   day) -> dict:
    """One VeteranCard per veteran in the candidate list, for the click-through."""
    fac = (pl.read_parquet(schema.REFERENCE / "va_facilities_nyc_hazard.parquet")
             .select("station_no", "name").rename({"name": "facility_name"}))
    wanted = sorted({c["veteran_id"] for c in candidates})
    co = (cohort.filter(pl.col("veteran_id").is_in(wanted))
                .join(fac, left_on="facility_id", right_on="station_no", how="left"))
    today = scores.filter((pl.col("date") == day) & pl.col("veteran_id").is_in(wanted))

    by_vet: dict[str, list[dict]] = {}
    for r in today.to_dicts():
        by_vet.setdefault(r["veteran_id"], []).append(r)
    planned: dict[str, list[str]] = {}
    tier_of: dict[str, str] = {}
    for c in candidates:
        planned.setdefault(c["veteran_id"], []).append(c["action"])
        tier_of[c["veteran_id"]] = c["tier"]

    out: dict[str, dict] = {}
    for v in co.to_dicts():
        vid = v["veteran_id"]
        rows = sorted(by_vet.get(vid, []), key=lambda r: -r["p_mean"])
        if not rows:
            continue
        needs = [api.NeedScore(
            need=r["need"], p_mean=round(r["p_mean"], 4), p_lo80=round(r["p_lo80"], 4),
            p_hi80=round(r["p_hi80"], 4), p_epistemic_share=round(r["p_epistemic_share"], 4),
            p_gap_lo=r.get("p_gap_lo"), p_gap_hi=r.get("p_gap_hi"),
            drivers=[r[f"driver_{i}"] for i in (1, 2, 3) if r.get(f"driver_{i}")],
            driver_contribs=[round(r[f"driver_{i}_contrib"], 4) for i in (1, 2, 3)
                             if r.get(f"driver_{i}")],
        ) for r in rows]
        tier = tier_of.get(vid, "everyday")
        notes = []
        if v["med_combo_raas_diuretic"]:
            notes.append("RAAS agent and a diuretic together, the combination CDC names")
        if v["med_renal_triple"]:
            notes.append("NSAID on top of that pair: the renal triple whammy")
        if v["med_cold_chain"]:
            notes.append("refrigerated medicine at home, so an outage is a clinical event")
        if v["med_controlled"]:
            notes.append("controlled: the retail emergency refill does not cover it")
        if v["med_narrow_ti"]:
            notes.append("narrow therapeutic index, so a missed dose matters quickly")
        if v["acb_score"] >= 3:
            notes.append(f"anticholinergic burden {v['acb_score']}, which blunts sweating")
        if v["mail_order_pharmacy"] and v["days_supply_remaining"] <= 14:
            notes.append(f"{v['days_supply_remaining']} days of supply left and it comes by mail")
        card = api.VeteranCard(
            veteran_id=vid, name_display=v["name_display"], age=int(v["age"]),
            modzcta=v["modzcta"], borough=v["borough"], facility_id=v["facility_id"],
            facility_name=v["facility_name"] or f"Station {v['facility_id']}", date=day,
            tier=tier,
            why_this_tier=why_tier(tier, needs, severity.load()),
            needs=needs,
            medications=api.MedicationFlags(
                n_active_meds=int(v["n_active_meds"]),
                thermoreg_score=round(float(v["med_thermoreg_score"]), 2),
                acb_score=int(v["acb_score"]),
                combo_raas_diuretic=bool(v["med_combo_raas_diuretic"]),
                renal_triple=bool(v["med_renal_triple"]),
                cold_chain=bool(v["med_cold_chain"]),
                controlled=bool(v["med_controlled"]),
                narrow_ti=bool(v["med_narrow_ti"]),
                mail_order_pharmacy=bool(v["mail_order_pharmacy"]),
                days_supply_remaining=int(v["days_supply_remaining"]),
                notes=notes,
            ),
            conditions=[label for col, label in CONDITION_LABELS.items() if v[col]],
            powered_equipment=v["powered_equipment"], caregiver=v["caregiver"],
            floor=v["floor"], evac_zone=int(v["evac_zone"]),
            planned_actions=sorted(set(planned.get(vid, []))),
        )
        out[vid] = card.model_dump(mode="json")
    return out


def build_messages(cohort: pl.DataFrame, candidates: list[dict]) -> dict:
    """One Message per candidate action.

    Every body carries the five elements that let a veteran tell this from a scam: the VA
    channel tag, a four-word phrase the caller reads back, the never-pay line, VSAFE and
    the Veterans Crisis Line. tests/test_ui_fixtures.py asserts all five on every row.
    """
    by_vet = {c["veteran_id"]: c for c in
              cohort.select("veteran_id", "caregiver", "caregiver_contact_consent").to_dicts()}
    out: dict[str, dict] = {}
    for c in candidates:
        v = by_vet.get(c["veteran_id"])
        if v is None:
            continue
        aid = c["action_id"]
        phrase = _phrase(aid)
        channel = CHANNEL_FOR_OWNER.get(c["owner"], "VEText")
        to_caregiver = (bool(v["caregiver_contact_consent"])
                        and str(v["caregiver"]).startswith("informal"))
        ask = MESSAGE_ASK.get(c["action"], "Your VA care team would like to reach you.")
        opener = ("This is the VA New York Harbor care team, with a message about the veteran "
                  "you care for." if to_caregiver
                  else "This is your VA New York Harbor care team.")
        body = (
            f"[VA {channel}] {opener}\n"
            f"{ask}\n"
            f"When we call we will say the words \u201c{phrase}\u201d so you know it is really "
            "us. Ask us to say them.\n"
            "The VA will never ask you to pay, wire money, or share bank details.\n"
            "Report a suspected scam to VSAFE at 833-388-7233.\n"
            "Veterans Crisis Line: dial 988, press 1."
        )
        msg = api.Message(
            message_id=c["message_id"] or f"msg-{aid}", action_id=aid,
            veteran_id=c["veteran_id"], channel=channel,
            addressed_to="caregiver" if to_caregiver else "veteran",
            verification_phrase=phrase, body=body,
            includes_never_pay_line=True, includes_vsafe=True, includes_crisis_line=True,
            scam_card_url="https://news.va.gov/vsafe/",
        )
        out[aid] = msg.model_dump(mode="json")
    return out

def build_report(scores: pl.DataFrame) -> dict:
    """ReportResponse for the Model report screen.

    `make report` assembles the whole thing -- recovery, reliability, decision quality, the
    fairness audit -- into report/report.json. That file is generated output and gitignored,
    so a clean clone has no eval run at all and `GET /report` answers with the rung and
    nothing else. Copy it into the fixtures when it is there, so the screen has the real run
    with the network off and on a machine that has never run the pipeline.

    When it is not there, fall back to what has actually been run: the rung, and decision
    quality from report/decision_quality.csv if `python -m leeward.eval.decision_quality`
    has produced it (mean realised harm averted per day, by K and strategy). Every other
    section stays empty and the screen says so; an empty fairness table means "not audited",
    never "passed".
    """
    from datetime import datetime

    full = ROOT / "report" / "report.json"
    if full.exists():
        return api.ReportResponse.model_validate_json(
            full.read_text(encoding="utf-8")).model_dump(mode="json")

    rows = []
    csv = ROOT / "report" / "decision_quality.csv"
    if csv.exists():
        tidy = pl.read_csv(csv)
        agg = (tidy.group_by("k", "strategy").agg(pl.col("harm_averted").mean())
                   .sort("k", "strategy"))
        rows = [api.DecisionQualityRow(k=int(r["k"]), strategy=r["strategy"],
                                       harm_averted=round(float(r["harm_averted"]), 3))
                for r in agg.to_dicts()]
    resp = api.ReportResponse(model_rung=int(scores["model_rung"][0]), decision_quality=rows,
                              generated_at=datetime.now().isoformat(timespec="minutes") if rows else None)
    return resp.model_dump(mode="json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="forecast window length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date", type=str, default=None, help="demo day, as a date")
    ap.add_argument("--day", type=int, default=None,
                    help="demo day, scenario-relative: 0 is the first day in hazards.parquet")
    args = ap.parse_args(argv)
    if args.date and args.day is not None:
        raise SystemExit("pass --date or --day, not both")

    cohort = schema.read("cohort")
    scores = schema.read("scores")
    actions = schema.read("actions")
    hazards = schema.read("hazards")
    # Default: the window the demo opens on, off the hazards table (leeward/demo.py), so a
    # rebuild lands on landfall week rather than on whatever day 0 happens to be. The API
    # answers an unasked `GET /forecast` with the same window, so offline and live agree.
    if args.date:
        day = date.fromisoformat(args.date)
    elif args.day is not None:
        scenario_days = sorted(hazards["date"].unique().to_list())
        if not 0 <= args.day < len(scenario_days):
            raise SystemExit(f"--day {args.day} is outside 0..{len(scenario_days) - 1}")
        day = scenario_days[args.day]
    else:
        day = demo.opening_date(hazards, schema.read("site_status"), window=args.days)
    if actions.filter(pl.col("date") == day).height == 0:
        raise SystemExit(f"actions.parquet has no rows on {day}; run `make score`, or pass "
                         "--date/--day for a day it does cover")
    dates = [day + timedelta(days=i) for i in range(args.days)]
    have = set(scores["date"].unique().to_list())
    dates = [d for d in dates if d in have]
    if not dates:
        raise SystemExit("scores.parquet does not cover the actions date; regenerate fixtures")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"UI fixtures for {dates[0]} .. {dates[-1]}  (model rung {scores['model_rung'][0]})")
    scenario_days = sorted(hazards["date"].unique().to_list())
    _dump(build_forecast(dates, scenario_days.index(dates[0])).model_dump(mode="json"),
          OUT / "forecast.json")
    _dump(build_scores(cohort, scores, dates), OUT / "scores.json")
    cands = build_candidates(cohort, scores, day, args.seed)
    _dump(cands, OUT / "actions_candidates.json")
    _dump(build_veterans(cohort, scores, cands["candidates"], day), OUT / "veterans.json")
    _dump(build_messages(cohort, cands["candidates"]), OUT / "messages.json")
    report = build_report(scores)
    _dump(report, OUT / "report.json")
    if report["decision_quality"] and not (ROOT / "report" / "report.json").exists():
        # The API serves report/report.json; give it the same partial report so the live
        # screen and the offline screen agree. `make report` overwrites this.
        _dump(report, ROOT / "report" / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

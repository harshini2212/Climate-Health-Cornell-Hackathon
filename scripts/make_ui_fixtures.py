"""Write the JSON the UI reads until the API exists, shaped exactly like leeward/api/schemas.py.

    python scripts/make_fixtures.py --start 2026-06-01 --days 120
    python -m leeward.ingest.hazards --scenario scenarios/sandy_then_heat.yaml
    python scripts/make_ui_fixtures.py

Outputs, all under ui/public/fixtures/ and committed (they are small):

    forecast.json             ForecastResponse: 7 days from the actions date, from
                              data/hazards.parquet + data/site_status.parquet
    scores.json               { date: { need: ScoresResponse } } for the same 7 days,
                              from data/scores.parquet aggregated per ZIP
    actions_candidates.json   a ranked candidate list from data/scores.parquet; the UI
                              cuts it at whatever capacity the slider asks for, so the
                              slider tells the real story before allocate.py lands

Every object is validated through the pydantic models before it is written, so the UI
is typed against the contract and not against whatever this script happened to emit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl

from leeward import schema
from leeward.api import schemas as api
from leeward.schema import ACTION_COST_UNIT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ui" / "public" / "fixtures"

#: Severity weights w_k, the same defaults decision/severity.py will carry.
SEVERITY = {"breathing": 3.0, "heat": 4.0, "mental": 4.0, "treatment_gap": 5.0, "access_loss": 3.0}

#: A plausible action for the need that dominates a veteran's day.
ACTION_FOR_NEED = {
    "breathing": "clean_air_room",
    "heat": "cooling_center_ride",
    "mental": "care_team_call",
    "treatment_gap": "early_refill",
    "access_loss": "alt_site_booking",
}
RATIONALE_FIND_OUT = "Three-minute check-in: wide interval, we do not know the AC or the floor. Re-score today."
RATIONALE_EVERYDAY = "Low risk this week: verified wellness text with the cooling-site list."
RATIONALE = {
    "breathing": "Clean-air room match before PM2.5 peaks; inhaler supply check today.",
    "heat": "Book the cooling-center ride for the first heat-alert day; do not just suggest it.",
    "mental": "Care-team call today; move Thursday's therapy to phone if the outage lands.",
    "treatment_gap": "Early refill now: supply runs out inside the forecast window.",
    "access_loss": "Pre-arrange the alternate site while the home station is down.",
}


def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=None, separators=(",", ":"), default=str) + "\n",
                    encoding="utf-8")
    print(f"  {path.relative_to(ROOT).as_posix():44s} {path.stat().st_size / 1e3:8.1f} kB")


def build_forecast(dates: list) -> api.ForecastResponse:
    hz = schema.read("hazards").filter(pl.col("date").is_in(dates))
    # FacilityStatus carries no date, so a site counts as down if it is down on any day
    # of the forecast window: "will this site be there when the veteran needs it?"
    site = (schema.read("site_status").filter(pl.col("date").is_in(dates))
                  .group_by("facility_id")
                  .agg(pl.col("site_down").any(), pl.col("evac_zone").first(),
                       pl.col("site_dependent_services").first())
                  .sort("facility_id"))
    fac = pl.read_parquet(schema.REFERENCE / "va_facilities_nyc_hazard.parquet")
    site = site.join(fac.select("station_no", "name", "lat", "lon"),
                     left_on="facility_id", right_on="station_no", how="left")
    zip_fields = [f for f in api.ZipHazard.model_fields]
    zips = [api.ZipHazard(**{k: r[k] for k in zip_fields}) for r in hz.select(zip_fields).to_dicts()]
    facilities = [api.FacilityStatus(**{k: r[k] for k in api.FacilityStatus.model_fields})
                  for r in site.to_dicts()]
    down = [f.name for f in facilities if f.site_down]
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
        headline += f". Site down: {', '.join(down)}"
    return api.ForecastResponse(scenario="sandy_then_heat", day=0, dates=dates, zips=zips,
                                facilities=facilities, headline=headline)


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
    """Rank every veteran by severity-weighted peak risk; keep enough to fill 100 calls."""
    r = np.random.default_rng(seed)
    today = scores.filter(pl.col("date") == day)
    w = pl.col("need").replace_strict(SEVERITY, return_dtype=pl.Float64)
    per_vet = (today.with_columns((pl.col("p_mean") * w).alias("weighted"))
                    .sort("weighted", descending=True)
                    .group_by("veteran_id", maintain_order=True)
                    .agg(pl.col("need").first().alias("top_need"),
                         pl.col("weighted").sum().alias("eha_raw"),
                         pl.col("p_mean").max().alias("peak"),
                         pl.col("p_epistemic_share").mean().alias("epi"),
                         pl.col("driver_1").first().alias("top_driver"))
                    .join(cohort.select("veteran_id", "name_display", "modzcta", "borough",
                                        "age", "n_chronic", "ckd_dialysis", "on_methadone_otp"),
                          on="veteran_id")
                    .sort("eha_raw", descending=True)
                    .head(600))
    rows = []
    for i, v in enumerate(per_vet.to_dicts()):
        tier = ("act_now" if v["peak"] >= 0.25 and v["epi"] < 0.4
                else "find_out" if v["epi"] >= 0.4 and v["peak"] >= 0.10
                else "self_serve" if v["peak"] >= 0.05 else "everyday")
        # Most people get one action; act-now veterans get a call plus their need's action.
        acts = [ACTION_FOR_NEED[v["top_need"]]]
        if tier == "act_now" and "care_team_call" not in acts:
            acts.append("care_team_call")
        elif tier == "find_out":
            acts = ["check_in_call"]
        elif tier == "everyday":
            acts = ["verified_text"]
        for j, action in enumerate(acts):
            aid = hashlib.sha1(f"{day}{v['veteran_id']}{action}".encode()).hexdigest()[:12]
            rows.append({
                "action_id": aid, "rank": i + 1, "veteran_id": v["veteran_id"],
                "name_display": v["name_display"], "modzcta": v["modzcta"],
                "borough": v["borough"], "action": action, "tier": tier,
                "eha": round(float(v["eha_raw"]) * (0.7 if j else 1.0) * 0.6, 3),
                "capacity_bucket": ACTION_COST_UNIT[action],
                "owner": ("pharmacist" if action == "pharmacist_med_review"
                          else "automated" if action == "verified_text" else "care_team"),
                "rationale": (RATIONALE_FIND_OUT if action == "check_in_call"
                              else RATIONALE_EVERYDAY if action == "verified_text"
                              else RATIONALE[v["top_need"]]),
                "top_driver": v["top_driver"], "message_id": f"msg-{aid}",
                "age": int(v["age"]), "n_chronic": int(v["n_chronic"]),
                "rand": float(r.random()),
            })
    for row in rows:  # every candidate must be a legal ActionRow
        api.ActionRow(**{k: row[k] for k in api.ActionRow.model_fields})
    return {"date": str(day), "model_rung": int(scores["model_rung"][0]),
            "n_panel": cohort.height, "candidates": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="forecast window length")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cohort = schema.read("cohort")
    scores = schema.read("scores")
    actions = schema.read("actions")
    day = actions["date"][0]
    dates = [day + timedelta(days=i) for i in range(args.days)]
    have = set(scores["date"].unique().to_list())
    dates = [d for d in dates if d in have]
    if not dates:
        raise SystemExit("scores.parquet does not cover the actions date; regenerate fixtures")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"UI fixtures for {dates[0]} .. {dates[-1]}  (model rung {scores['model_rung'][0]})")
    _dump(build_forecast(dates).model_dump(mode="json"), OUT / "forecast.json")
    _dump(build_scores(cohort, scores, dates), OUT / "scores.json")
    _dump(build_candidates(cohort, scores, day, args.seed), OUT / "actions_candidates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

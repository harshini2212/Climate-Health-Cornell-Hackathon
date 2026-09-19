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
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from leeward import schema
from leeward.api import schemas as api
from leeward.schema import DEFAULT_CAPACITY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ui" / "public" / "fixtures"

#: Severity weights w_k, the same defaults decision/severity.py will carry.
SEVERITY = {"breathing": 3.0, "heat": 4.0, "mental": 4.0, "treatment_gap": 5.0, "access_loss": 3.0}



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
    """The real allocator's list at the slider's maximum (100 calls), so the client can cut
    it at any smaller capacity with the same rules and still match what /actions would say.
    Baselines need age, n_chronic and a seeded random key per row, so those ride along."""
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
            "rationale": v["rationale"], "top_driver": v["top_driver"],
            "message_id": v["message_id"],
            "age": int(v["age"]), "n_chronic": int(v["n_chronic"]), "rand": float(r.random()),
        })
    for row in rows:  # every candidate must be a legal ActionRow
        api.ActionRow(**{k: row[k] for k in api.ActionRow.model_fields})
    return {"date": str(day), "model_rung": int(scores["model_rung"][0]),
            "n_panel": cohort.height, "candidates": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="forecast window length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date", type=str, default=None,
                    help="demo day (default: first day in actions.parquet). For Sandy use "
                         "2026-08-03, landfall day")
    args = ap.parse_args(argv)

    cohort = schema.read("cohort")
    scores = schema.read("scores")
    actions = schema.read("actions")
    day = date.fromisoformat(args.date) if args.date else actions["date"][0]
    if actions.filter(pl.col("date") == day).height == 0:
        raise SystemExit(f"actions.parquet has no rows on {day}")
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

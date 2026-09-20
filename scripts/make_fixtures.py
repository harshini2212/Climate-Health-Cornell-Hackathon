#!/usr/bin/env python3
"""Generate correctly-shaped fake data for every contract table.

This is what unblocks six parallel lanes. The API can serve, the UI can render, the
decision layer can allocate and the eval harness can plot -- all before a single real
cohort exists. Each lane then replaces one fixture with the real thing, and because the
shape never changes, nothing downstream breaks.

The geography is NOT fake: ZIPs, boroughs, facilities and hazard exposure come from
`data/reference/`, so the map is the real NYC map from minute one and a wrong join shows
up immediately rather than at hour twelve.

    python scripts/make_fixtures.py               # 500 veterans, 30 days
    python scripts/make_fixtures.py --veterans 2000 --days 60

Everything is seeded. Same command, same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import date, timedelta

import numpy as np
import polars as pl

from leeward import schema
from leeward.decision import tau
from leeward.schema import (
    ACTION_COST_UNIT,
    CAREGIVER,
    DEFAULT_CAPACITY,
    DEPLOYMENT_ERAS,
    FLOORS,
    INCOME_BANDS,
    NEEDS,
    POWERED_EQUIPMENT,
    REFERENCE,
    TIERS,
)

FIRST = ["Walter", "James", "Robert", "Linda", "Michael", "Carmen", "Dennis", "Patricia",
         "Luis", "Gloria", "Anthony", "Denise", "Raymond", "Yolanda", "Frank", "Marta"]
LAST = ["Reyes", "Okafor", "Delgado", "Nguyen", "Brooks", "Castellano", "Whitfield",
        "Ferreira", "Kowalski", "Ramsey", "Baptiste", "Ortiz", "Coleman", "Vasquez"]

DRIVER_PHRASES = [
    "dialysis at a station that is closed",
    "oxygen concentrator during a forecast outage",
    "ground-floor unit in a high-flood ZIP",
    "hydrochlorothiazide and lisinopril on a 96F day",
    "nine days of medication left, mail delivery at risk",
    "no caregiver and an evacuation order for this zone",
    "COPD on a PM2.5 above 150 day",
    "owns an air conditioner, utility shutoff risk in this ZIP",
    "anticholinergic burden of 4 during a heat alert",
    "insulin at home with an 80 percent outage forecast",
    "controlled substance: retail emergency refill does not cover it",
]

RATIONALES = [
    "Call today: treatment gap likely before Thursday and the interval is narrow.",
    "Three-minute check-in: we do not know the floor or the AC status.",
    "Early refill now: supply runs out inside the forecast window.",
    "Book the ride, do not suggest it: low assets and a transport barrier.",
    "Pharmacist review before the heat wave: additive medication combination.",
    "Pre-arrange the alternate dialysis site while the station is down.",
]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _ref(name: str) -> pl.DataFrame:
    path = REFERENCE / f"{name}.parquet"
    if not path.exists():
        raise SystemExit(f"missing {path}. Run: python scripts/fetch_sources.py")
    return pl.read_parquet(path)


def _geography() -> tuple[list[str], dict[str, str], pl.DataFrame, pl.DataFrame]:
    """Real NYC ZIPs with their real borough, hazard exposure and facilities."""
    mz = _ref("nyc_modzcta")
    evac = _ref("evac_zone_by_modzcta")
    storm = _ref("stormwater_by_modzcta")
    hvi = _ref("hvi_by_zcta").with_columns(pl.col("zcta").alias("modzcta"))
    fac = _ref("va_facilities_nyc_hazard")

    # MODZCTA carries no borough column; derive it from the nearest facility's borough is
    # wrong, so use the ZIP prefix ranges NYC actually uses.
    def boro(z: str) -> str:
        n = int(z)
        if 10001 <= n <= 10282:
            return "Manhattan"
        if 10301 <= n <= 10314:
            return "Staten Island"
        if 10451 <= n <= 10475:
            return "Bronx"
        if 11201 <= n <= 11256:
            return "Brooklyn"
        return "Queens"

    zips = (mz.select("modzcta")
              .join(evac.select("modzcta", "evac_zone_min"), on="modzcta", how="left")
              .join(storm, on="modzcta", how="left")
              .join(hvi.select("modzcta", "hvi"), on="modzcta", how="left")
              .with_columns([
                  pl.col("modzcta").map_elements(boro, return_dtype=pl.Utf8).alias("borough"),
                  pl.col("evac_zone_min").fill_null(0).cast(pl.Int32).alias("evac_zone"),
                  pl.col("stormwater_flooded_frac").fill_null(0.0),
                  pl.col("hvi").fill_null(3).cast(pl.Int32),
              ]).drop("evac_zone_min"))
    return zips["modzcta"].to_list(), dict(zip(zips["modzcta"], zips["borough"], strict=False)), zips, fac


def make_cohort(n: int, seed: int, zips: pl.DataFrame, fac: pl.DataFrame) -> pl.DataFrame:
    r = _rng(seed)
    z = zips.sample(n, with_replacement=True, seed=seed)
    stations = fac["station_no"].to_list()

    age = np.clip(r.normal(68, 13, n), 22, 99).astype(int)
    older = age >= 65
    hvi = z["hvi"].to_numpy()
    # Higher-HVI ZIPs get worse social and resource draws, matching the PLACES gradient.
    hvi_lift = (hvi - 1) / 4.0

    copd = r.random(n) < (0.04 + 0.05 * hvi_lift + 0.04 * older)
    n_meds = r.poisson(3.2 + 2.0 * older, n).clip(0, 25)
    thermo = np.round(r.gamma(1.4, 0.75, n) * (n_meds > 0), 2)
    acb = r.poisson(0.7 + 0.6 * older, n).clip(0, 12)
    raas_diur = r.random(n) < 0.16 + 0.10 * older
    mail = r.random(n) < 0.80
    supply = np.where(mail, r.integers(0, 91, n), r.integers(0, 31, n))

    ids = [f"SYN-{i:06d}" for i in range(n)]
    names = [f"{FIRST[r.integers(len(FIRST))]} {LAST[r.integers(len(LAST))]}" for _ in range(n)]

    def pick(opts, p=None):
        return [opts[i] for i in r.choice(len(opts), n, p=p)]

    df = pl.DataFrame({
        "veteran_id": ids,
        "name_display": names,
        "age": age.astype(np.int32),
        "sex": pick(["M", "F"], [0.88, 0.12]),
        "race": pick(["White", "Black", "Asian", "Other"], [0.55, 0.27, 0.08, 0.10]),
        "ethnicity": pick(["Hispanic", "Non-Hispanic"], [0.26, 0.74]),
        "modzcta": z["modzcta"],
        "borough": z["borough"],
        "facility_id": [stations[i] for i in r.integers(len(stations), size=n)],
        "copd": copd,
        "asthma": r.random(n) < 0.09,
        "chf": r.random(n) < 0.06 + 0.06 * older,
        "diabetes": r.random(n) < 0.18 + 0.10 * hvi_lift,
        "ckd_dialysis": r.random(n) < 0.015,
        "active_cancer_tx": r.random(n) < 0.03,
        "ptsd": r.random(n) < 0.22,
        "depression": r.random(n) < 0.19,
        "pact_presumptive": r.random(n) < 0.17,
        "n_chronic": r.poisson(2.4 + 1.2 * older, n).clip(0, 15).astype(np.int32),
        "er_visits_12m": r.poisson(0.6, n).astype(np.int32),
        "missed_refills_12m": r.poisson(0.5, n).astype(np.int32),
        "missed_appts_12m": r.poisson(0.9, n).astype(np.int32),
        "med_rxcuis": [[str(x) for x in r.integers(1000, 999999, size=int(k))]
                       for k in n_meds],
        "va_drug_classes": [["CV700", "CV800"][: int(min(k, 2))] for k in n_meds],
        "n_active_meds": n_meds.astype(np.int32),
        "med_thermoreg_score": thermo,
        "acb_score": acb.astype(np.int32),
        "med_combo_raas_diuretic": raas_diur,
        "med_renal_triple": raas_diur & (r.random(n) < 0.12),
        "med_cold_chain": r.random(n) < 0.09,
        "med_controlled": r.random(n) < 0.10,
        "med_narrow_ti": r.random(n) < 0.11,
        "mail_order_pharmacy": mail,
        "days_supply_remaining": supply.astype(np.int32),
        "deployment_era": pick(DEPLOYMENT_ERAS, [0.34, 0.27, 0.24, 0.15]),
        "burn_pit_years": np.round(r.lognormal(-0.6, 0.8, n) * (r.random(n) < 0.45), 2),
        "ptsd_severity": r.integers(0, 5, n).astype(np.int32),
        "home_ac": r.random(n) < (0.95 - 0.19 * hvi_lift),
        "floor": pick(FLOORS, [0.06, 0.25, 0.69]),
        "on_methadone_otp": r.random(n) < 0.012,
        "powered_equipment": pick(POWERED_EQUIPMENT, [0.86, 0.06, 0.01, 0.04, 0.02, 0.01]),
        "lives_alone": r.random(n) < 0.34,
        "mobility_impaired": r.random(n) < (0.09 + 0.10 * hvi_lift),
        "caregiver": pick(CAREGIVER, [0.38, 0.37, 0.19, 0.06]),
        "caregiver_contact_consent": r.random(n) < 0.72,
        "income_band": pick(INCOME_BANDS, [0.34, 0.45, 0.21]),
        "low_assets": r.random(n) < (0.05 + 0.12 * hvi_lift),
        "transport_barrier": r.random(n) < (0.06 + 0.11 * hvi_lift),
        "evac_zone": z["evac_zone"],
        "stormwater_flooded_frac": z["stormwater_flooded_frac"],
        "hvi": z["hvi"],
        "consent_partner_check": r.random(n) < 0.63,
        "consent_ride": r.random(n) < 0.71,
        "consent_housing": r.random(n) < 0.48,
    })
    # Every synthetic column needs its flag. In fixtures, everything is synthetic.
    flags = {f"{c.name}_synthetic": pl.lit(True)
             for c in schema.TABLES["cohort"].columns if c.synthetic}
    return df.with_columns(**flags)


def make_hazards(zips: pl.DataFrame, days: list[date], seed: int) -> pl.DataFrame:
    """A scripted Sandy-then-heat shape so the UI has something with a story in it."""
    r = _rng(seed + 1)
    rows = []
    n_z = zips.height
    surge_day, heat_start = len(days) // 2, len(days) // 2 + 3
    for d_i, d in enumerate(days):
        surge = d_i in (surge_day - 1, surge_day, surge_day + 1)
        heat = heat_start <= d_i <= heat_start + 2
        base_t = 74 + 12 * np.sin(d_i / 6) + r.normal(0, 3, n_z)
        temp = base_t + (22 if heat else 0)
        ez = zips["evac_zone"].to_numpy()
        in_surge = surge & (ez > 0) & (ez <= 2)
        outage = np.where(in_surge, r.uniform(0.4, 0.9, n_z), r.uniform(0, 0.03, n_z))
        rows.append(pl.DataFrame({
            "modzcta": zips["modzcta"],
            "date": [d] * n_z,
            "heat_index_max_f": np.round(temp, 1),
            "hot_day": temp >= 82,
            "heat_alert": np.full(n_z, bool(heat)),
            "pm25": np.round(np.clip(r.gamma(2, 4, n_z), 0, 400), 1),
            "smoke_alert": np.full(n_z, False),
            "flood_watch": np.full(n_z, bool(surge)),
            "flood_warning": in_surge,
            "flash_flood_emergency": np.full(n_z, False),
            "surge_ft": np.where(in_surge, r.uniform(5, 10, n_z),
                                 np.zeros(n_z)).round(1),
            "evac_zone_ordered": np.full(n_z, 2 if surge else 0, dtype=np.int32),
            "floodnet_trip": in_surge & (r.random(n_z) < 0.3),
            "stormwater_flooded_frac": zips["stormwater_flooded_frac"],
            "outage_frac": outage.round(2),
            "mail_delivery_disrupted": in_surge | (outage > 0.5),
        }))
    return pl.concat(rows)


def make_site_status(fac: pl.DataFrame, days: list[date]) -> pl.DataFrame:
    down_from = len(days) // 2
    rows = []
    for d_i, d in enumerate(days):
        rows.append(pl.DataFrame({
            "facility_id": fac["station_no"],
            "date": [d] * fac.height,
            # Station 630 sits in evacuation zone 1; it is the one that goes down.
            "site_down": [(d_i >= down_from and s == "630") for s in fac["station_no"]],
            "evac_zone": fac["evac_zone"].cast(pl.Int32),
            "site_dependent_services": fac["site_dependent_services"],
        }))
    return pl.concat(rows)


def make_scores(cohort: pl.DataFrame, days: list[date], seed: int) -> pl.DataFrame:
    r = _rng(seed + 2)
    n, T, K = cohort.height, len(days), len(NEEDS)
    vids = np.repeat(cohort["veteran_id"].to_numpy(), T * K)
    # Build dates as a Python list so polars infers pl.Date, not Object.
    dates = [d for _ in range(n) for d in days for _ in NEEDS]
    needs = NEEDS * (n * T)

    total = n * T * K
    p = np.clip(r.beta(1.6, 14, total), 0.001, 0.98)
    width = np.clip(r.gamma(2, 0.035, total), 0.01, 0.5)
    lo = np.clip(p - width / 2, 0.0005, 0.99)
    hi = np.clip(p + width / 2, lo + 0.001, 0.999)

    di = r.integers(0, len(DRIVER_PHRASES), (total, 3))
    return pl.DataFrame({
        "veteran_id": vids,
        "date": pl.Series("date", dates, dtype=pl.Date),
        "need": needs,
        "p_mean": p.round(4),
        "p_lo80": lo.round(4),
        "p_hi80": hi.round(4),
        "p_epistemic_share": np.clip(r.beta(2, 4, total), 0, 1).round(3),
        "driver_1": [DRIVER_PHRASES[i] for i in di[:, 0]],
        "driver_2": [DRIVER_PHRASES[i] for i in di[:, 1]],
        "driver_3": [DRIVER_PHRASES[i] for i in di[:, 2]],
        "driver_1_contrib": r.normal(0.8, 0.3, total).round(3),
        "driver_2_contrib": r.normal(0.5, 0.2, total).round(3),
        "driver_3_contrib": r.normal(0.3, 0.15, total).round(3),
        "model_rung": np.zeros(total, dtype=np.int32),
    })


def make_outcomes(scores: pl.DataFrame, seed: int) -> pl.DataFrame:
    r = _rng(seed + 3)
    p = scores["p_mean"].to_numpy()
    return scores.select("veteran_id", "date", "need").with_columns(
        pl.Series("y", (r.random(len(p)) < p).astype(np.int32)))


def make_actions(cohort: pl.DataFrame, scores: pl.DataFrame, day: date, seed: int) -> pl.DataFrame:
    """A greedy fill that obeys the same rules `allocate.py` will have to obey.

    Sorted by EHA, capped per capacity bucket, one action per veteran unless act_now.
    The fixture is not just shaped right, it is *behaviourally* right -- so the guardrails
    in tests/test_guardrails.py pass against it, and a real allocator that breaks them
    fails loudly rather than quietly replacing a correct fixture with a wrong table.

    `day` is the **do-by day**: one work list, capped at one day's capacity. Each row's
    `lead_days` comes from tau.yaml, so its risk day is `day + lead_days` -- inside the
    scored window, because `main()` puts the fixture day in the middle of it.
    """
    r = _rng(seed + 4)
    todays = (scores.filter(pl.col("date") == day)
                    .group_by("veteran_id")
                    .agg(pl.col("p_mean").max().alias("peak"),
                         pl.col("p_epistemic_share").mean().alias("epi"),
                         pl.col("driver_1").first().alias("top_driver"))
                    .sort("peak", descending=True)
                    .head(400))

    peak = todays["peak"].to_numpy()
    epi = todays["epi"].to_numpy()
    tier = np.where(peak > 0.18, TIERS[0], np.where(epi > 0.45, TIERS[1], TIERS[2]))
    eha = (peak * r.uniform(2, 5, todays.height)).round(3)

    choices = ["care_team_call", "early_refill", "check_in_call", "cooling_center_ride",
               "pharmacist_med_review", "alt_site_booking", "verified_text",
               "controlled_substance_bridge"]
    act = r.choice(choices, todays.height)

    cand = (todays.with_columns([pl.Series("tier", tier), pl.Series("eha", eha),
                                 pl.Series("action", act)])
                  .sort("eha", descending=True))

    remaining = dict(DEFAULT_CAPACITY)
    per_vet: dict[str, int] = {}
    kept = []
    for row in cand.to_dicts():
        bucket = ACTION_COST_UNIT[row["action"]]
        limit = 3 if row["tier"] == TIERS[0] else 1
        if per_vet.get(row["veteran_id"], 0) >= limit:
            continue
        if remaining[bucket] <= 0:
            continue
        remaining[bucket] -= 1
        per_vet[row["veteran_id"]] = per_vet.get(row["veteran_id"], 0) + 1
        kept.append(row)

    n = len(kept)
    acts = [k["action"] for k in kept]
    vets = [k["veteran_id"] for k in kept]
    leads = tau.leads()
    aid = [hashlib.sha1(f"{day}{v}{a}".encode()).hexdigest()[:12] for v, a in zip(vets, acts, strict=False)]
    owners = ["pharmacist" if a == "pharmacist_med_review"
              else "automated" if a == "verified_text" else "care_team" for a in acts]
    return pl.DataFrame({
        "action_id": aid,
        "date": pl.Series("date", [day] * n, dtype=pl.Date),
        "veteran_id": vets,
        "action": acts,
        "tier": [k["tier"] for k in kept],
        "eha": [k["eha"] for k in kept],
        "rank": np.arange(1, n + 1, dtype=np.int32),
        "lead_days": pl.Series("lead_days", [leads[a] for a in acts], dtype=pl.Int32),
        "capacity_bucket": [ACTION_COST_UNIT[a] for a in acts],
        "rationale": [RATIONALES[i] for i in r.integers(0, len(RATIONALES), n)],
        "owner": owners,
        "message_id": [f"msg-{x}" for x in aid],
    })


def make_outcome_log(actions: pl.DataFrame, seed: int) -> pl.DataFrame:
    r = _rng(seed + 5)
    a = actions.head(25)
    n = a.height
    return pl.DataFrame({
        "action_id": a["action_id"],
        "veteran_id": a["veteran_id"],
        "date": a["date"],
        "done": r.random(n) < 0.8,
        "reached": r.random(n) < 0.65,
        "need_occurred": r.random(n) < 0.2,
        "partner_ack": r.random(n) < 0.3,
        "logged_by": ["fixture"] * n,
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--veterans", type=int, default=500)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=str, default="2026-07-01")
    args = ap.parse_args()

    _, _, zips, fac = _geography()
    start = date.fromisoformat(args.start)
    days = [start + timedelta(days=i) for i in range(args.days)]

    cohort = make_cohort(args.veterans, args.seed, zips, fac)
    hazards = make_hazards(zips, days, args.seed)
    site = make_site_status(fac, days)
    scores = make_scores(cohort, days, args.seed)
    outcomes = make_outcomes(scores, args.seed)
    actions = make_actions(cohort, scores, days[len(days) // 2], args.seed)
    log = make_outcome_log(actions, args.seed)

    for df, name in [(cohort, "cohort"), (hazards, "hazards"), (site, "site_status"),
                     (outcomes, "outcomes"), (scores, "scores"),
                     (actions, "actions"), (log, "outcome_log")]:
        path = schema.write(df, name)
        print(f"  {name:14s} {df.height:>8,} rows  {path.stat().st_size / 1e6:6.2f} MB")

    print(f"\nFixtures written for {args.veterans} veterans x {args.days} days. "
          "Every table validates against leeward/schema.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

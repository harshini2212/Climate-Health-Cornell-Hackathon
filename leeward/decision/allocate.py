"""Today's ranked action list, cut at the care team's real capacity (SPEC §7.4).

    allocate(scores, cohort, capacity, group_floor=None) -> the `actions` table
    total_eha(actions) -> float

Each candidate is one (veteran, action) pair on one day. It uses one unit of its capacity
bucket (`schema.ACTION_COST_UNIT`) and one of the veteran's slots for the day: one, or three
if the veteran is Act-now. Every day in `scores` is allocated separately, each with the full
`capacity`.

Greedy by EHA per unit cost. Every action costs exactly one unit of its own bucket, and the
buckets have no exchange rate -- a call is not a ride -- so EHA per unit cost is EHA, and
the per-bucket cap does the rest. Highest EHA goes first; a candidate is skipped when its
bucket is full or its veteran has no slot left.

An Act-now veteran's actions do not add up naively: two actions that each prevent 60% of a
heat illness prevent 84% together, not 120%. So each of their actions is valued on the risk
left after all of their higher-valued actions (`_values`). A single-slot veteran's value is
the plain SPEC §7.3 EHA. Either way the `eha` column is at most the harm the action truly
averts, and `total_eha` never claims harm that is not there.

More capacity never averts less harm, because every value is fixed before anything is
chosen. Raise one bucket by one and the two runs agree until the first candidate the old run
turned away for room. From there they differ by one swap at a time -- the new run gains that
candidate, which may cost its veteran a later one, which frees a slot for a later one still
-- and each swap is worth no more than the one before, so the gains cover the losses.
(Re-valuing an Act-now veteran's actions *as they are chosen* breaks this: a seeded search
found capacity increases that lowered the total by up to 7%.)

`compare()` is `allocate()` plus baselines: what the same team, with the same capacity and the
same candidate actions valued the same way, averts when it works the veterans in some other
order -- oldest first, most chronic conditions first, a seeded shuffle. Only who goes first
changes; each veteran's own actions are still tried best first, so the gap to `allocate()` is
what risk-ranking the veterans is worth. The shared work is done once and only the greedy pass
repeats, which is what keeps `POST /actions` inside its 300 ms.

`group_floor={"borough": 0.1}` reserves floor(0.1 x capacity) of every bucket for each
borough. Reserved slots are filled greedily first; whatever a group cannot use goes back to
everyone. A floor is a constraint, so it can cost harm averted, and the argument above does
not cover it.

A bucket missing from `capacity` has no capacity. A caller holding a partial dict (the API)
merges it onto `DEFAULT_CAPACITY` first.

    python -m leeward.decision.allocate [--date 2026-07-16]      # -> data/actions.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections.abc import Mapping
from datetime import date as Date

import numpy as np
import polars as pl

from leeward import schema
from leeward.decision import eha, severity, tiers
from leeward.decision import tau as tau_table
from leeward.schema import ACTION_COST_UNIT, DEFAULT_CAPACITY, NEEDS

CHECK_IN = "check_in_call"
ACT_NOW_SLOTS = 3
#: An action worth less than this averts nothing. Do not spend a slot on it.
EPS = 1e-12
BUCKETS = sorted(set(ACTION_COST_UNIT.values()))

#: Who does it. Medication actions go to the pharmacist, who decides; Leeward changes nothing.
OWNER = {
    "pharmacist_med_review": "pharmacist",
    "early_refill": "pharmacist",
    "switch_to_local_pickup": "pharmacist",
    "controlled_substance_bridge": "pharmacist",
    "verified_text": "automated",
    "assign_buddy": "partner",
    "heap_application": "partner",
}

LEAD = {
    "care_team_call": "Call today",
    "backup_power_plan": "Call to confirm a backup-power plan for their equipment",
    "cold_chain_plan": "Call about keeping refrigerated medication cold",
    "early_refill": "Ask the pharmacist to approve an early refill",
    "switch_to_local_pickup": "Ask the pharmacist to move this fill from mail to local pickup",
    "controlled_substance_bridge": ("Flag for the pharmacist a controlled-substance bridge, "
                                    "which the retail emergency refill does not cover"),
    "pharmacist_med_review": ("Flag for the VA clinical pharmacist, who decides whether "
                              "anything changes"),
    "cooling_center_ride": "Book a ride to a cooling center",
    "clean_air_room": "Book a ride to a clean-air room",
    "alt_site_booking": "Book the alternate {service} site",
    "evacuation_assist": "Arrange evacuation help",
    "verified_text": "Send the verified text",
    "assign_buddy": "Assign a buddy check",
    "heap_application": "Start a HEAP cooling-assistance application",
}

NEED_PHRASE = {
    "breathing": "breathing trouble",
    "heat": "heat illness",
    "mental": "a mental-health crisis",
    "treatment_gap": "a gap in treatment",
    "access_loss": "losing access to care",
}

OUT_SCHEMA = {c.name: c.dtype for c in schema.TABLES["actions"].columns} | {
    "top_need": pl.Utf8,      # the need this action does the most for
    "top_driver": pl.Utf8,    # that need's first driver, for ActionRow.top_driver
}


def allocate(
    scores: pl.DataFrame,
    cohort: pl.DataFrame,
    capacity: dict[str, int],
    group_floor: dict[str, float] | None = None,
    *,
    date: Date | None = None,
    weights: dict[str, float] | None = None,
    tau: dict[str, dict[str, float]] | None = None,
) -> pl.DataFrame:
    """The `actions` table for every day in `scores` (or only `date`), cut at `capacity`.

    `weights` and `tau` default to severity.yaml and tau.yaml.
    """
    return compare(scores, cohort, capacity, group_floor, date=date, weights=weights,
                   tau=tau)[0]


def compare(
    scores: pl.DataFrame,
    cohort: pl.DataFrame,
    capacity: dict[str, int],
    group_floor: dict[str, float] | None = None,
    *,
    rank_by: Mapping[str, str] | None = None,
    date: Date | None = None,
    weights: dict[str, float] | None = None,
    tau: dict[str, dict[str, float]] | None = None,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """`allocate()`'s table, and the total EHA of each baseline in `rank_by`.

    `rank_by` maps a baseline's name to a numeric cohort column; that baseline works the
    veterans from the highest value of the column down (ties in veteran order, so the answer
    does not depend on risk). The total is summed over every day allocated, like `total_eha`.
    """
    weights = weights if weights is not None else severity.load()
    table = tau if tau is not None else tau_table.load()
    cap = _check_capacity(capacity)
    floor = _check_floor(group_floor or {}, cohort)
    rank_by = dict(rank_by or {})
    missing = {n: c for n, c in rank_by.items() if c not in cohort.columns}
    if missing:
        raise ValueError(f"rank_by names cohort columns that do not exist: {missing}")
    if date is not None:
        scores = scores.filter(pl.col("date") == date)
    if scores.height == 0:
        return _empty(), dict.fromkeys(rank_by, 0.0)

    wide = eha.needs_wide(scores)
    unknown = wide.join(cohort, on="veteran_id", how="anti")["veteran_id"].unique()
    if unknown.len():
        raise ValueError(f"{unknown.len()} scored veterans are not in the cohort, "
                         f"e.g. {unknown.sort()[0]!r}")
    cols = ["veteran_id", *dict.fromkeys([*eha.COHORT_COLUMNS, *floor, *rank_by.values()])]
    frame = (wide.join(tiers.assign(scores, weights), on=["veteran_id", "date"])
                 .join(cohort.select(cols), on="veteran_id")
                 .sort("date", "veteran_id"))
    days = [_allocate_day(day, cap, floor, weights, table, rank_by)
            for _, day in frame.group_by("date", maintain_order=True)]
    totals = {n: sum(t[n] for _, t in days) for n in rank_by}
    return pl.concat([a for a, _ in days]), totals


def total_eha(actions: pl.DataFrame) -> float:
    """Harm averted by the whole list: the sum of `eha`, marginal values included."""
    return float(actions["eha"].sum()) if actions.height else 0.0


def _allocate_day(day: pl.DataFrame, cap: dict[str, int], floor: dict[str, float],
                  weights: dict[str, float], table: dict[str, dict[str, float]],
                  rank_by: Mapping[str, str]) -> tuple[pl.DataFrame, dict[str, float]]:
    n = day.height
    vids = day["veteran_id"].to_list()
    tier = day["tier"].to_numpy()
    p = day.select([f"p_mean_{k}" for k in NEEDS]).to_numpy()
    share = day.select([f"p_epistemic_share_{k}" for k in NEEDS]).to_numpy()
    w = severity.vector(weights)

    prevent, T = tau_table.matrix(table)
    actions = [*prevent, CHECK_IN]
    ci = len(prevent)                                   # column of the check-in
    bucket = [ACTION_COST_UNIT[a] for a in actions]
    open_ = day.select([eha.ELIGIBLE[a].alias(a) for a in actions]).to_numpy().astype(bool)
    open_ &= (tier != "everyday")[:, None]
    open_[:, ci] &= tier == "find_out"

    slots = np.where(tier == "act_now", ACT_NOW_SLOTS, 1)
    value, top = _values(p * w, w * eha.epistemic_var(p, share), T, open_, slots)

    i_idx, a_idx = np.nonzero(open_ & (value > EPS))
    groups = {col: day[col].to_list() for col in floor}

    def pick(order_by: np.ndarray | None) -> set[tuple[int, int]]:
        """One greedy pass over the candidates: highest value first, or -- for a baseline --
        veterans from the highest `order_by` down, each one's own actions best first.
        (value, veteran, action) breaks ties the same way every run."""
        if order_by is None:
            order = np.lexsort((a_idx, i_idx, -value[i_idx, a_idx]))
        else:
            order = np.lexsort((a_idx, -value[i_idx, a_idx], i_idx, -order_by[i_idx]))
        queue = list(zip(i_idx[order].tolist(), a_idx[order].tolist(), strict=True))

        used = np.zeros(n, dtype=int)
        remaining = dict(cap)
        quota = {(col, g, b): math.floor(f * cap[b] + 1e-9)
                 for col, f in floor.items() for g in set(groups[col]) for b in cap}
        taken: set[tuple[int, int]] = set()

        def run(reserved_only: bool) -> None:
            for i, a in queue:
                b = bucket[a]
                if used[i] >= slots[i] or remaining[b] <= 0 or (i, a) in taken:
                    continue
                mine = [(col, groups[col][i], b) for col in floor]
                if reserved_only and not any(quota[q] > 0 for q in mine):
                    continue
                taken.add((i, a))
                used[i] += 1
                remaining[b] -= 1
                for q in mine:
                    quota[q] = max(0, quota[q] - 1)

        if floor:
            run(reserved_only=True)
        run(reserved_only=False)
        return taken

    taken = pick(None)
    totals = {name: float(sum(value[i, a] for i, a in pick(day[col].to_numpy().astype(float))))
              for name, col in rank_by.items()}
    if not taken:
        return _empty(), totals
    chosen = [(i, a, float(value[i, a]), int(top[i, a])) for i, a in sorted(taken)]

    lo = day.select([f"p_lo80_{k}" for k in NEEDS]).to_numpy()
    hi = day.select([f"p_hi80_{k}" for k in NEEDS]).to_numpy()
    drivers = day.select([f"driver_1_{k}" for k in NEEDS]).rows()
    service = day.select(
        pl.when("ckd_dialysis").then(pl.lit("dialysis"))
          .when("active_cancer_tx").then(pl.lit("infusion"))
          .otherwise(pl.lit("opioid treatment program"))
    ).to_series().to_list()
    on = day["date"][0]

    rows = []
    for i, a, value_, k in chosen:
        act, need = actions[a], NEEDS[k]
        aid = hashlib.sha1(f"{on}|{vids[i]}|{act}".encode()).hexdigest()[:12]
        rows.append({
            "action_id": aid, "date": on, "veteran_id": vids[i], "action": act,
            "tier": tier[i], "eha": value_, "rank": 0, "capacity_bucket": bucket[a],
            "rationale": _rationale(act, need, p[i, k], lo[i, k], hi[i, k], share[i, k],
                                    drivers[i][k], service[i]),
            "owner": OWNER.get(act, "care_team"), "message_id": f"msg-{aid}",
            "top_need": need, "top_driver": drivers[i][k],
        })
    actions_df = (pl.DataFrame(rows, schema=OUT_SCHEMA)
                    .sort(["eha", "veteran_id", "action"], descending=[True, False, False])
                    .with_columns(rank=pl.int_range(1, pl.len() + 1, dtype=pl.Int32)))
    return actions_df, totals


def _values(risk: np.ndarray, info: np.ndarray, T: np.ndarray, open_: np.ndarray,
            slots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EHA per (veteran, action), fixed before anything is chosen, and the need it helps most.

    `risk` is weighted risk w*p (n x K); `info` is weighted epistemic variance (n x K), the
    check-in's value, which sits in the last column. A one-slot veteran's value is the
    standalone EHA. A multi-slot veteran's actions are taken in descending standalone EHA and
    each is valued on the risk left after all of the ones above it, whether or not those are
    chosen -- so the value never depends on capacity, and a veteran's chosen actions never sum
    to more than the harm they carry.
    """
    gain = np.einsum("nk,ak->nak", risk, T)             # per-need EHA of every action
    gain = np.concatenate([gain, info[:, None, :]], axis=1)
    gain[~open_] = 0.0
    for i in np.nonzero(slots > 1)[0]:
        standalone = gain[i].sum(axis=1)
        left = risk[i].copy()
        for a in np.lexsort((np.arange(len(T)), -standalone[: len(T)])):
            if standalone[a] <= EPS:
                break
            gain[i, a] = left * T[a]
            left = left * (1 - T[a])
    return gain.sum(axis=2), gain.argmax(axis=2)


def _rationale(action: str, need: str, p: float, lo: float, hi: float, share: float,
               driver: str | None, service: str) -> str:
    """One sentence a care-team member can read aloud."""
    if action == CHECK_IN:
        return (f"Three-minute check-in to find out: {p:.0%} chance of {NEED_PHRASE[need]}, "
                f"and {share:.0%} of the uncertainty is what we do not know about them.")
    why = f", driven by {driver}" if driver else ""
    return (f"{LEAD[action].format(service=service)}: {p:.0%} chance of {NEED_PHRASE[need]} "
            f"(80% interval {lo:.0%}-{hi:.0%}){why}.")


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=OUT_SCHEMA)


def _check_capacity(capacity: dict[str, int]) -> dict[str, int]:
    unknown = sorted(set(capacity) - set(BUCKETS))
    if unknown:
        raise ValueError(f"unknown capacity buckets {unknown}; buckets are {BUCKETS}")
    out = {b: int(capacity.get(b, 0)) for b in BUCKETS}
    negative = {b: v for b, v in out.items() if v < 0}
    if negative:
        raise ValueError(f"capacity cannot be negative: {negative}")
    return out


def _check_floor(group_floor: dict[str, float], cohort: pl.DataFrame) -> dict[str, float]:
    for col, f in group_floor.items():
        if col not in cohort.columns:
            raise ValueError(f"group_floor names {col!r}, which is not a cohort column")
        if not 0 <= f <= 1:
            raise ValueError(f"group_floor {col}={f} must be a share between 0 and 1")
        n = cohort[col].n_unique()
        if n * f > 1 + 1e-9:
            raise ValueError(f"group_floor {col}={f} reserves {n} groups x {f:.0%} = "
                             f"{n * f:.0%} of every bucket; the most it can be is {1 / n:.3f}")
    return dict(group_floor)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="scores + cohort -> data/actions.parquet, "
                                             "cut at DEFAULT_CAPACITY")
    ap.add_argument("--date", type=Date.fromisoformat, default=None,
                    help="allocate one day (default: every day in scores)")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    actions = allocate(schema.read("scores"), schema.read("cohort"), DEFAULT_CAPACITY,
                       date=args.date)
    path = schema.write(actions, "actions")
    print(f"  actions  {actions.height:,} rows over {actions['date'].n_unique()} days, "
          f"total EHA {total_eha(actions):,.1f}, {time.perf_counter() - t0:.1f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

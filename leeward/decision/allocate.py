"""Today's ranked action list, cut at the care team's real capacity (SPEC §7.4).

    allocate(scores, cohort, capacity, group_floor=None) -> the `actions` table
    total_eha(actions) -> float

Each candidate is one (veteran, action) pair for the risk of one day. It uses one unit of its
capacity bucket (`schema.ACTION_COST_UNIT`) and one of the veteran's slots for the day.

**An action has a day it must be done by, which is not the day the risk lands.** If the surge
hits Wednesday, the alternate dialysis site is booked by Monday; offering it on Wednesday
averts nothing, because there is nothing left to book. So every action carries a lead time
from `tau.yaml`, its do-by day is `risk day - lead_days`, and it competes for the capacity of
*that* day. `actions.date` is the do-by day -- the day it goes on the care team's list -- and
`date + lead_days` is the day the risk lands. Both are marks the week board can draw.

An action whose do-by day falls before the first day this call is planning for is already
behind the team and is not offered at all: on Wednesday, an alternate site for Wednesday's
surge is not a hard action, it is an impossible one. `compare()` returns how many of those
the team would have taken, which is a real number worth showing -- part of what the forecast
asks for on any given day could only have been done days ago.

Each do-by day gets the full `capacity`, drawn from every risk day that maps onto it. The
veteran's slot limit, though, is counted twice over -- once per risk day and once per do-by
day, both one or three. Only the first is the SPEC §7.4 rule, and it is the one `_values`
prices against; the second is the care team's time with one person in one day. Dropping
either lets a veteran's reported EHA climb past the harm they carry, or puts nine things
beside one name on a Monday. Because those budgets cross days, the days are worked as one
queue rather than one at a time (`_fill`).

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
found capacity increases that lowered the total by up to 7%.) Lead times do not disturb the
argument. They add capacities -- a bucket per do-by day instead of one, and the second slot
budget -- but every one of them is fixed before anything is chosen, which is all the argument
rests on. `test_scheduled_days_still_never_avert_less_with_more_capacity` checks it with the
days genuinely interleaved.

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
import itertools
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date as Date
from datetime import timedelta

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

INSTRUCTION = {
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

#: The whole vocabulary the wall board gets for *why*. A need is a thing that may happen to
#: someone; a condition, a medicine and a service are things that are true of them. Only the
#: first kind may sit beside a de-identified handle in a room other people walk through.
NEED_HEADLINE = {
    "breathing": "Breathing trouble",
    "heat": "Heat illness",
    "mental": "A mental-health crisis",
    "treatment_gap": "A gap in treatment",
    "access_loss": "Loss of access to care",
}

#: What the tier asks of whoever is reading the board, and by when. Deliberately says
#: nothing about *which* action: the card shows that above the headline already.
TIER_URGENCY = {
    "act_now": "act today",
    "find_out": "confirm today",
    "self_serve": "this week",
    "everyday": "no action today",
}

OUT_SCHEMA = {c.name: c.dtype for c in schema.TABLES["actions"].columns} | {
    "risk_date": pl.Date,     # date + lead_days: the day the risk lands, for the week ribbon
    "headline": pl.Utf8,      # the card-safe line the wall board shows (see `_headline`)
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
    lead: dict[str, int] | None = None,
) -> pl.DataFrame:
    """The `actions` table for every do-by day `scores` can cover (or only `date`).

    `weights`, `tau` and `lead` default to severity.yaml and tau.yaml.
    """
    return compare(scores, cohort, capacity, group_floor, date=date, weights=weights,
                   tau=tau, lead=lead)[0]


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
    lead: dict[str, int] | None = None,
) -> tuple[pl.DataFrame, dict[str, float], int]:
    """`allocate()`'s table, each baseline's total EHA, and the too-late count.

    `rank_by` maps a baseline's name to a numeric cohort column; that baseline works the
    veterans from the highest value of the column down (ties in veteran order, so the answer
    does not depend on risk). The total is summed over every day allocated, like `total_eha`.

    `date` names one **do-by day** -- the day the care team is working. Reaching it needs the
    risk of the days after it too, so `scores` is read from `date` forward as far as the
    longest lead time, and only the actions that land on `date` come back.

    The third return is how many actions the team would have taken on a do-by day earlier
    than the first day being planned -- `date`, or the first day in `scores`. They are not in
    the table, because offering an action that can no longer be done is worse than none.
    """
    weights = weights if weights is not None else severity.load()
    table = tau if tau is not None else tau_table.load()
    lead = _check_lead(lead if lead is not None else tau_table.lead_days(), table)
    cap = _check_capacity(capacity)
    floor = _check_floor(group_floor or {}, cohort)
    rank_by = dict(rank_by or {})
    missing = {n: c for n, c in rank_by.items() if c not in cohort.columns}
    if missing:
        raise ValueError(f"rank_by names cohort columns that do not exist: {missing}")
    horizon = max(lead.values(), default=0)
    if date is not None:
        scores = scores.filter(pl.col("date").is_between(date, date + timedelta(days=horizon)))
    if scores.height == 0:
        return _empty(), dict.fromkeys(rank_by, 0.0), 0

    wide = eha.needs_wide(scores)
    unknown = wide.join(cohort, on="veteran_id", how="anti")["veteran_id"].unique()
    if unknown.len():
        raise ValueError(f"{unknown.len()} scored veterans are not in the cohort, "
                         f"e.g. {unknown.sort()[0]!r}")
    cols = ["veteran_id", *dict.fromkeys([*eha.COHORT_COLUMNS, *floor, *rank_by.values()])]
    frame = (wide.join(tiers.assign(scores, weights), on=["veteran_id", "date"])
                 .join(cohort.select(cols), on="veteran_id")
                 .sort("date", "veteran_id"))

    # The first and last day being planned. A do-by day before `arrived` is already behind the
    # team. When `date` is given it is both, even if it is itself unscored -- asking for a day
    # must not push that day's own work into the too-late pile.
    arrived = date if date is not None else frame["date"].min()
    last = date if date is not None else frame["date"].max()
    panel = _panel(frame, table, lead, floor, rank_by)
    n = len(panel.index)
    priced = [_price(d, day, panel, weights)
              for d, (_, day) in enumerate(frame.group_by("date", maintain_order=True))]
    # Blocks past `last` belong to a day this call is not planning. Blocks before the forecast
    # arrived are allocated separately, in their own counterfactual day with its own full
    # capacity: that is what "the list the team would have been given" means, and it must not
    # spend any of the capacity the offered days are entitled to.
    offered = [b for day in priced for b in day.blocks if arrived <= b.do_by <= last]
    gone = [b for day in priced for b in day.blocks if b.do_by < arrived]

    n_too_late = len(_fill(gone, n, cap, floor, panel.groups, {})[0]) if gone else 0
    if not offered:
        return _empty(), dict.fromkeys(rank_by, 0.0), n_too_late
    chosen, totals = _fill(offered, n, cap, floor, panel.groups, panel.order)
    tables = [_rows(do_by, list(rows), priced, panel, lead)
              for do_by, rows in itertools.groupby(chosen, key=lambda c: c[0])]
    return (pl.concat(tables) if tables else _empty()), totals, n_too_late


def total_eha(actions: pl.DataFrame) -> float:
    """Harm averted by the whole list: the sum of `eha`, marginal values included."""
    return float(actions["eha"].sum()) if actions.height else 0.0


@dataclass(frozen=True)
class _Block:
    """One risk day's candidates that share a do-by day: the unit a pool is built from.

    Every action with the same lead time moves the same distance, so a risk day splits into
    one block per distinct lead. Keeping them as arrays rather than rows is what lets a pool
    be assembled by concatenation instead of a Python loop over 50,000 candidates.
    """
    day: int                  #: index into the `priced` list -- which risk day this came from
    do_by: Date
    row: np.ndarray           #: veteran's row within that day
    act: np.ndarray           #: index into `_Priced.actions`
    value: np.ndarray
    veteran: np.ndarray       #: veteran's index across the whole window
    bucket: np.ndarray        #: index into BUCKETS
    slots: np.ndarray         #: the veteran's slot limit on that risk day


@dataclass(frozen=True)
class _Panel:
    """Everything that is true of the veterans rather than of a day, worked out once.

    Eligibility and which service a veteran depends on come from the cohort, and the cohort
    does not change between Monday and Wednesday. Re-deriving them per risk day was free
    when there was only ever one; with a lead-time window it is the same 15 polars
    expressions over the whole panel, three times over.
    """
    index: dict[str, int]                    #: veteran_id -> row in this panel
    actions: list[str]
    T: np.ndarray                            #: tau, one row per prevention action
    eligible: np.ndarray                     #: panel x action, from eha.ELIGIBLE
    service: list[str]
    lead_of: np.ndarray
    bucket_of: np.ndarray
    groups: dict[str, list]                  #: group_floor column -> label per veteran
    order: dict[str, np.ndarray]             #: baseline name -> sort key per veteran


def _panel(frame: pl.DataFrame, table: dict[str, dict[str, float]], lead: dict[str, int],
           floor: dict[str, float], rank_by: Mapping[str, str]) -> _Panel:
    rows = frame.unique(subset="veteran_id", keep="first").sort("veteran_id")
    prevent, T = tau_table.matrix(table)
    actions = [*prevent, CHECK_IN]
    return _Panel(
        index={v: i for i, v in enumerate(rows["veteran_id"].to_list())},
        actions=actions, T=T,
        eligible=rows.select([eha.ELIGIBLE[a].alias(a) for a in actions]).to_numpy().astype(bool),
        service=rows.select(
            pl.when("ckd_dialysis").then(pl.lit("dialysis"))
              .when("active_cancer_tx").then(pl.lit("infusion"))
              .otherwise(pl.lit("opioid treatment program"))).to_series().to_list(),
        lead_of=np.array([lead[a] for a in actions]),
        bucket_of=np.array([BUCKETS.index(ACTION_COST_UNIT[a]) for a in actions]),
        groups={col: rows[col].to_list() for col in floor},
        order={name: rows[col].to_numpy().astype(float) for name, col in rank_by.items()})


@dataclass(frozen=True)
class _Priced:
    """One risk day, valued. Everything the chosen rows need to be written out."""
    date: Date
    veteran: np.ndarray       #: each row's index into the panel
    vids: list[str]
    tier: np.ndarray
    value: np.ndarray
    top: np.ndarray
    p: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    share: np.ndarray
    drivers: list[tuple]
    blocks: list[_Block]


def _price(which: int, day: pl.DataFrame, panel: _Panel,
           weights: dict[str, float]) -> _Priced:
    """Value every (veteran, action) pair for one risk day, and post each to its do-by day."""
    vids = day["veteran_id"].to_list()
    veteran = np.array([panel.index[v] for v in vids])
    tier = day["tier"].to_numpy()
    p = day.select([f"p_mean_{k}" for k in NEEDS]).to_numpy()
    share = day.select([f"p_epistemic_share_{k}" for k in NEEDS]).to_numpy()
    w = severity.vector(weights)

    ci = len(panel.actions) - 1                         # column of the check-in
    open_ = panel.eligible[veteran]
    open_ = open_ & (tier != "everyday")[:, None]
    open_[:, ci] &= tier == "find_out"

    slots = np.where(tier == "act_now", ACT_NOW_SLOTS, 1)
    value, top = _values(p * w, w * eha.epistemic_var(p, share), panel.T, open_, slots)

    on = day["date"][0]
    i_idx, a_idx = np.nonzero(open_ & (value > EPS))
    ahead = panel.lead_of[a_idx]
    blocks = []
    for days_ahead in sorted(set(ahead.tolist())):
        m = ahead == days_ahead
        blocks.append(_Block(day=which, do_by=on - timedelta(days=int(days_ahead)),
                             row=i_idx[m], act=a_idx[m], value=value[i_idx[m], a_idx[m]],
                             veteran=veteran[i_idx[m]], bucket=panel.bucket_of[a_idx[m]],
                             slots=slots[i_idx[m]]))
    return _Priced(date=on, veteran=veteran, vids=vids, tier=tier, value=value, top=top, p=p,
                   lo=day.select([f"p_lo80_{k}" for k in NEEDS]).to_numpy(),
                   hi=day.select([f"p_hi80_{k}" for k in NEEDS]).to_numpy(),
                   share=share,
                   drivers=day.select([f"driver_1_{k}" for k in NEEDS]).rows(),
                   blocks=blocks)


def _fill(blocks: list[_Block], n_panel: int, cap: dict[str, int], floor: dict[str, float],
          groups: dict[str, list], order: Mapping[str, np.ndarray],
          ) -> tuple[list[tuple[Date, int, int, int]], dict[str, float]]:
    """One greedy pass over every candidate in the window, highest EHA first.

    Three budgets, and a candidate needs room in all of them:

    * one unit of its capacity bucket **on its do-by day** -- the care team's Monday has
      40 calls in it whether the risk is Monday's or Wednesday's;
    * one of the veteran's slots **on its risk day** -- one, or three if they are Act-now.
      This is the budget `_values` prices against, so relaxing it would let a veteran's
      reported EHA climb past the harm they actually carry;
    * one of the veteran's slots **on its do-by day** -- the care team's time with that
      person on the day they pick up the phone, which does not stretch because two
      different days are at stake.

    The two slot budgets cross days, so the days are worked as one queue rather than one
    pool at a time. Every value is still fixed before anything is chosen, which is what the
    capacity-monotonicity argument in the module docstring needs.
    """
    days = sorted({b.do_by for b in blocks})
    at = {d: i for i, d in enumerate(days)}
    day_of = np.concatenate([np.full(b.row.shape, b.day) for b in blocks])
    do_by = np.concatenate([np.full(b.row.shape, at[b.do_by]) for b in blocks])
    row = np.concatenate([b.row for b in blocks])
    act = np.concatenate([b.act for b in blocks])
    value = np.concatenate([b.value for b in blocks])
    veteran = np.concatenate([b.veteran for b in blocks])
    bucket = np.concatenate([b.bucket for b in blocks])
    slot_of = np.concatenate([b.slots for b in blocks])

    nd, nr = len(days), int(day_of.max()) + 1
    slots_do = np.zeros(n_panel * nd, dtype=int)
    slots_risk = np.zeros(n_panel * nr, dtype=int)
    np.maximum.at(slots_do, veteran * nd + do_by, slot_of)
    np.maximum.at(slots_risk, veteran * nr + day_of, slot_of)

    # (veteran, day) and (bucket, day) pairs flattened, so the hot loop below indexes plain
    # Python lists: one greedy pass touches every candidate several times.
    key_do = (veteran * nd + do_by).tolist()
    key_risk = (veteran * nr + day_of).tolist()
    key_cap = (bucket * nd + do_by).tolist()
    slots_do, slots_risk = slots_do.tolist(), slots_risk.tolist()
    start = [cap[b] for b in BUCKETS for _ in range(nd)]
    mine = [[(col, groups[col][v], b, d) for col in floor]
            for v, b, d in zip(veteran.tolist(), [BUCKETS[i] for i in bucket.tolist()],
                               do_by.tolist(), strict=True)] if floor else None

    def pick(order_by: np.ndarray | None) -> list[int]:
        """One greedy pass over the candidates: highest value first, or -- for a baseline --
        veterans from the highest `order_by` down, each one's own actions best first.
        (value, veteran, action) breaks ties the same way every run."""
        if order_by is None:
            queue = np.lexsort((act, day_of, veteran, -value))
        else:
            queue = np.lexsort((act, day_of, -value, veteran, -order_by[veteran]))
        queue = queue.tolist()

        used_do = [0] * (n_panel * nd)
        used_risk = [0] * (n_panel * nr)
        remaining = list(start)
        quota = {(col, g, b, d): math.floor(f * cap[b] + 1e-9)
                 for col, f in floor.items() for g in set(groups[col])
                 for b in cap for d in range(nd)}
        taken: list[int] = []
        seen = bytearray(len(queue))

        def run(reserved_only: bool) -> None:
            for c in queue:
                kd, kr, kc = key_do[c], key_risk[c], key_cap[c]
                if (seen[c] or used_do[kd] >= slots_do[kd]
                        or used_risk[kr] >= slots_risk[kr] or remaining[kc] <= 0):
                    continue
                reserved = mine[c] if mine else ()
                if reserved_only and not any(quota[q] > 0 for q in reserved):
                    continue
                seen[c] = 1
                taken.append(c)
                used_do[kd] += 1
                used_risk[kr] += 1
                remaining[kc] -= 1
                for q in reserved:
                    quota[q] = max(0, quota[q] - 1)

        if floor:
            run(reserved_only=True)
        run(reserved_only=False)
        return taken

    taken = pick(None)
    totals = {name: float(value[pick(by)].sum()) for name, by in order.items()}
    chosen = sorted((days[do_by[c]], int(day_of[c]), int(row[c]), int(act[c])) for c in taken)
    return chosen, totals


def _rows(do_by: Date, chosen: list[tuple[Date, int, int, int]], priced: list[_Priced],
          panel: _Panel, lead: dict[str, int]) -> pl.DataFrame:
    """One do-by day's chosen candidates, written out as `actions` rows and ranked."""
    rows = []
    for _, d, i, a in chosen:
        day = priced[d]
        act = panel.actions[a]
        k = int(day.top[i, a])
        need = NEEDS[k]
        aid = hashlib.sha1(f"{do_by}|{day.vids[i]}|{act}".encode()).hexdigest()[:12]
        rows.append({
            "action_id": aid, "date": do_by, "veteran_id": day.vids[i], "action": act,
            "lead_days": lead[act], "tier": day.tier[i], "eha": float(day.value[i, a]),
            "rank": 0, "capacity_bucket": ACTION_COST_UNIT[act],
            "rationale": _rationale(act, need, day.p[i, k], day.lo[i, k], day.hi[i, k],
                                    day.share[i, k], day.drivers[i][k],
                                    panel.service[day.veteran[i]]),
            "headline": _headline(need, day.p[i, k], day.tier[i]),
            "owner": OWNER.get(act, "care_team"), "message_id": f"msg-{aid}",
            "risk_date": day.date, "top_need": need, "top_driver": day.drivers[i][k],
        })
    return (pl.DataFrame(rows, schema=OUT_SCHEMA)
              .sort(["eha", "veteran_id", "action"], descending=[True, False, False])
              .with_columns(rank=pl.int_range(1, pl.len() + 1, dtype=pl.Int32)))


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


def _headline(need: str, p: float, tier: str) -> str:
    """The one line a wall-mounted board may show beside a de-identified handle.

    Urgency and timing, and nothing a passer-by could turn back into a chart: no condition,
    no medicine, no service, no name. "A gap in treatment, 34% chance -- act today".

    `_rationale` below is the other half of the same row and keeps all three, because the
    care-team drill-down is a private screen and cannot be acted on without them. Splitting
    them is the point: weakening `rationale` to fit the board would leave the person who
    picks up the phone with a sentence they cannot use.
    """
    return f"{NEED_HEADLINE[need]}, {p:.0%} chance — {TIER_URGENCY[tier]}"


def _rationale(action: str, need: str, p: float, lo: float, hi: float, share: float,
               driver: str | None, service: str) -> str:
    """One sentence a care-team member can read aloud. Names the service, the medicine and
    the driver on purpose -- so it is a drill-down line, never a board line (`_headline`)."""
    if action == CHECK_IN:
        return (f"Three-minute check-in to find out: {p:.0%} chance of {NEED_PHRASE[need]}, "
                f"and {share:.0%} of the uncertainty is what we do not know about them.")
    why = f", driven by {driver}" if driver else ""
    return (f"{INSTRUCTION[action].format(service=service)}: {p:.0%} chance of {NEED_PHRASE[need]} "
            f"(80% interval {lo:.0%}-{hi:.0%}){why}.")


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=OUT_SCHEMA)


def _check_lead(lead: dict[str, int], table: dict[str, dict[str, float]]) -> dict[str, int]:
    """Every action that can be offered needs a lead time. Missing is not zero."""
    need = {*table, CHECK_IN}
    missing = sorted(need - set(lead))
    if missing:
        raise ValueError(f"no lead time for {missing}; every offerable action needs one, "
                         "because a missing lead would silently schedule it for today")
    bad = {a: lead[a] for a in need if not isinstance(lead[a], int) or lead[a] < 0}
    if bad:
        raise ValueError(f"lead times must be a whole number of days, not {bad}")
    return {a: lead[a] for a in need}


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
                    help="allocate one do-by day (default: every day scores can cover)")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    actions, _, too_late = compare(schema.read("scores"), schema.read("cohort"),
                                   DEFAULT_CAPACITY, date=args.date)
    path = schema.write(actions, "actions")
    print(f"  actions  {actions.height:,} rows over {actions['date'].n_unique()} do-by days, "
          f"total EHA {total_eha(actions):,.1f}, {time.perf_counter() - t0:.1f}s -> {path}")
    if too_late:
        print(f"           {too_late:,} more were already too late to book when the "
              f"forecast arrived")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

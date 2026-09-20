"""A week's ranked action lists, cut at the care team's real capacity (SPEC §7.4).

    allocate(scores, cohort, capacity, group_floor=None) -> the `actions` table
    total_eha(actions) -> float

**An action has a day it must be done by, and it is not the day the risk lands.** If the
surge hits Wednesday, the alternate dialysis site has to be booked Monday; by Wednesday that
booking averts nothing. `tau.yaml` gives every action a lead time in days beside its tau, and
the do-by day is `risk day - lead_days`. That day, not the risk day, is the one the action
spends a unit of capacity on -- so Monday's forty calls cover Monday's own risk, Wednesday's
bookings and Saturday's refills together, which is what makes this a schedule rather than
seven daily lists. `date` in the `actions` table is the do-by day and `lead_days` rides
beside it, so the risk day is `date + lead_days`.

An action whose do-by day is before `as_of` -- the day the forecast arrived, the first day in
`scores` -- is not offered at all, because no amount of capacity can reach a day that has
gone. `compare()` returns how many, counted the honest way: not every candidate that missed
its window, but the work a team would actually have put on those days. It is a real number
worth saying out loud -- nine actions were already too late to book when the forecast came
in -- and for a single work day it reads as the work that had to happen before today.

Each candidate is one (veteran, action, risk day) triple. It uses one unit of its capacity
bucket (`schema.ACTION_COST_UNIT`) and one of the veteran's slots **on its do-by day**: one,
or three if the veteran is Act-now on the day the risk lands. Every do-by day is allocated
separately, each with the full `capacity`; no two do-by days share anything, which is what
lets `POST /actions` answer for one work day without allocating the whole week.

**Nobody gets a second action until everyone has been offered a first, Act-now veterans go
first inside each of those rounds, and EHA ranks within that** (SPEC §7.4). Three keys, in
that order, and each one earns its place:

*Round.* An Act-now veteran gets up to three slots. Offering all three before a Find-out
veteran is offered one is not priority, it is monopoly -- and it is expensive, because a
veteran's second and third actions are valued on the risk their first one leaves behind
(`_values`), so they are worth a fraction of somebody else's untouched first. Filling the
scarce buckets that way costs 17% of the harm this list averts. Taking the rounds in order
costs 6% and serves the same Act-now veterans.

*Band.* Within a round, `tiers.BAND` order: Act-now, then Find-out, then Self-serve, then
Everyday. Before this existed the order was pure greedy-by-EHA, and a Self-serve veteran
with broad moderate risk outranked an Act-now veteran with one sharp risk for the same call
-- so the tier badge said "call today" for people the list never called. Together with the
round, this is the promise the badge makes: no Self-serve veteran holds a scarce slot while
an Act-now veteran who could have used it holds nothing at all.

The band is a property of the **veteran on the work day**, not of the candidate: a veteran
is Act-now today if any of the risk days being worked for them today makes them Act-now, the
same "most urgent of the risk days" rule that sets how many slots they get. That is not a
detail. Banding each candidate separately would stop the greedy running in value order
*within a veteran*, and then a cheap Act-now action could take the slot a far more valuable
Self-serve one would have used, so raising a bucket could lower the total. Keyed this way it
cannot: round and band are both constant or monotone in value within a veteran, so a
veteran's own actions are still tried best first (see the monotonicity argument below).

*EHA.* Every action costs exactly one unit of its own bucket, and the buckets have no
exchange rate -- a call is not a ride -- so EHA per unit cost is EHA, and the per-bucket cap
does the rest. A candidate is skipped when its bucket is full or its veteran has no slot
left that day.

A veteran's actions do not add up naively: two actions that each prevent 60% of a heat
illness prevent 84% together, not 120%. So each action is valued on the risk left after all
of that veteran's higher-valued actions for the same risk day (`_values`), whether or not
those are chosen. That chain runs for every veteran, not just the Act-now ones, because the
schedule can hand a single-slot veteran one action on Monday and another on Wednesday for
the same Wednesday risk; standalone values would then claim more harm than the veteran
carries. The `eha` column is a floor on what the action truly averts, never a ceiling, and
`total_eha` never claims harm that is not there.

More capacity never averts less harm, because every value is fixed before anything is
chosen and do-by days do not compete. Raise one bucket by one and the two runs agree until
the first candidate the old run turned away for room. From there they differ by one swap at
a time: the new run gains that candidate, which costs its veteran one slot and so may cost
that **same veteran** a later action, which frees a unit of that action's bucket, which
gains a candidate for someone else, and so on. Every loss in the chain is paired with the
gain immediately above it and belongs to the same veteran, where the order is descending
EHA -- so each pair is worth zero or more and the gains cover the losses. That argument is
informal, so it was also searched: 5,400 capacity increases over 600 seeded Act-now-heavy
instances with the hazard rules firing, zero violations. (Re-valuing a veteran's actions *as
they are chosen* breaks this: the same kind of search found capacity increases that lowered
the total by up to 7%. So does banding each candidate rather than each veteran, for the
reason given above.)

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
from datetime import timedelta

import numpy as np
import polars as pl

from leeward import schema
from leeward.decision import eha, severity, tiers
from leeward.decision import rules as rule_table
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

#: The per-need fields the output rows read back out of the frame, one column per need.
_PER_NEED = ("p_mean", "p_lo80", "p_hi80", "p_epistemic_share", "driver_1")

OUT_SCHEMA = {c.name: c.dtype for c in schema.TABLES["actions"].columns} | {
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
    as_of: Date | None = None,
    hazards: pl.DataFrame | None = None,
    site_status: pl.DataFrame | None = None,
    rules: dict[str, rule_table.Rule] | None = None,
) -> pl.DataFrame:
    """The `actions` table for every do-by day `scores` reaches (or only `date`).

    `weights`, `tau`, `lead` and `rules` default to severity.yaml, tau.yaml and
    act_now.yaml. `hazards` and `site_status` switch on the four Act-now rules; without
    them the tiers come from the scores alone and no hazard rule can fire.
    """
    return compare(scores, cohort, capacity, group_floor, date=date, weights=weights,
                   tau=tau, lead=lead, as_of=as_of, hazards=hazards,
                   site_status=site_status, rules=rules)[0]


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
    as_of: Date | None = None,
    hazards: pl.DataFrame | None = None,
    site_status: pl.DataFrame | None = None,
    rules: dict[str, rule_table.Rule] | None = None,
) -> tuple[pl.DataFrame, dict[str, float], int]:
    """`allocate()`'s table, each baseline's total EHA, and the count that arrived too late.

    `date` picks one **do-by day**: the work list for that day, drawn from the risk days it
    can still reach (`date` through `date + max lead`). It is not a filter on the risk day,
    because the whole point is that the two differ.

    `as_of` is the day the forecast arrived; actions whose do-by day falls before it are not
    offered and are counted into the third return value. It defaults to `date`, or to the
    first day in `scores`.

    `rank_by` maps a baseline's name to a numeric cohort column; that baseline works the
    veterans from the highest value of the column down (ties in veteran order, so the answer
    does not depend on risk). The total is summed over every do-by day allocated, like
    `total_eha`. Baselines allocate under the **same tier band** as Leeward, because the band
    is the team's policy rather than a ranking: only who goes first may differ, or the gap
    would be measuring two changes at once instead of what risk-ranking is worth.
    """
    weights = weights if weights is not None else severity.load()
    table = tau if tau is not None else tau_table.load()
    # Once per call, not once per reader: `tiers.assign` needs the conditions and `_rows`
    # needs the sentences, and act_now.yaml carries a paragraph per rule.
    if rules is None and (hazards is not None or site_status is not None):
        rules = rule_table.load()
    cap = _check_capacity(capacity)
    floor = _check_floor(group_floor or {}, cohort)
    rank_by = dict(rank_by or {})
    missing = {n: c for n, c in rank_by.items() if c not in cohort.columns}
    if missing:
        raise ValueError(f"rank_by names cohort columns that do not exist: {missing}")

    prevent, T = tau_table.matrix(table)
    act_names = [*prevent, CHECK_IN]
    lead_of = _check_lead(lead if lead is not None else tau_table.leads(), act_names)
    horizon = int(lead_of.max(initial=0))

    if date is not None:
        scores = scores.filter(pl.col("date").is_between(date, date + timedelta(days=horizon)))
        as_of = date if as_of is None else as_of
    if scores.height == 0:
        return _empty(), dict.fromkeys(rank_by, 0.0), 0

    wide = eha.needs_wide(scores)
    unknown = wide.join(cohort, on="veteran_id", how="anti")["veteran_id"].unique()
    if unknown.len():
        raise ValueError(f"{unknown.len()} scored veterans are not in the cohort, "
                         f"e.g. {unknown.sort()[0]!r}")
    cols = ["veteran_id", *dict.fromkeys([*eha.COHORT_COLUMNS, *floor, *rank_by.values()])]
    graded = tiers.assign(scores, weights, cohort=cohort, hazards=hazards,
                          site_status=site_status, rules=rules)
    frame = (wide.join(graded, on=["veteran_id", "date"])
                 .join(cohort.select(cols), on="veteran_id")
                 .sort("date", "veteran_id")
                 .with_row_index("_unit")
                 .with_columns(_vet=(pl.col("veteran_id").rank("dense") - 1).cast(pl.Int64)))
    if as_of is None:
        as_of = frame["date"].min()

    cand = _candidates(frame, T, act_names, lead_of, weights)
    if date is not None:
        # Days after `date` belong to other work days and are not this one's business; days
        # before it still are, because they are what this day is too late for.
        in_reach = cand["do_by"] <= date.toordinal()
        cand = {k: v[in_reach] for k, v in cand.items()}

    slots = np.where(frame["tier"].to_numpy() == "act_now", ACT_NOW_SLOTS, 1)
    tier_band = frame["tier"].replace_strict(tiers.BAND, return_dtype=pl.Int64).to_numpy()
    vet = frame["_vet"].to_numpy()
    groups = {col: frame[col].to_list() for col in floor}
    orders = {name: frame[col].to_numpy().astype(float) for name, col in rank_by.items()}

    # Do-by days that have already gone are allocated too, but only to be counted: what a
    # team would have had to do before the forecast reached them is the honest size of
    # "too late", not the raw number of candidates that missed their window.
    want = date.toordinal() if date is not None else None
    gone = as_of.toordinal()
    chosen: list[dict[str, np.ndarray]] = []
    totals = dict.fromkeys(rank_by, 0.0)
    n_vets = int(vet.max()) + 1
    n_too_late = 0
    for day in _split_by_day(cand):
        on = int(day["do_by"][0])
        keep = on == want if want is not None else on >= gone
        taken, day_totals = _allocate_do_by_day(day, cap, floor, act_names, slots, tier_band,
                                                vet, groups, orders if keep else {}, n_vets)
        if taken is None:
            continue
        if keep:
            chosen.append(taken)
            for name, value in day_totals.items():
                totals[name] += value
        elif on < gone:
            n_too_late += len(taken["unit"])
    if not chosen:
        return _empty(), totals, n_too_late
    return _rows(chosen, frame, act_names, lead_of, rules), totals, n_too_late


def total_eha(actions: pl.DataFrame) -> float:
    """Harm averted by the whole list: the sum of `eha`, marginal values included."""
    return float(actions["eha"].sum()) if actions.height else 0.0


# --------------------------------------------------------------------------- #
# Valuing every candidate, once, before anything is chosen
# --------------------------------------------------------------------------- #

def _candidates(frame: pl.DataFrame, T: np.ndarray, act_names: list[str],
                lead_of: np.ndarray, weights: dict[str, float]) -> dict[str, np.ndarray]:
    """Every (veteran, action, risk day) worth spending a slot on, with its do-by day.

    Valued a risk day at a time, because a veteran's actions are valued against each other
    on the risk they share (`_values`). Which day the action is *done* on only decides which
    day's capacity it competes for, and that is settled afterwards.
    """
    w = severity.vector(weights)
    ci = len(T)                                         # column of the check-in
    unit, action, value, top, do_by = [], [], [], [], []
    for (day,), rows in frame.group_by("date", maintain_order=True):
        tier = rows["tier"].to_numpy()
        p = rows.select([f"p_mean_{k}" for k in NEEDS]).to_numpy()
        share = rows.select([f"p_epistemic_share_{k}" for k in NEEDS]).to_numpy()
        open_ = rows.select([eha.ELIGIBLE[a].alias(a) for a in act_names]).to_numpy().astype(bool)
        open_ &= (tier != "everyday")[:, None]
        open_[:, ci] &= tier == "find_out"

        val, best = _values(p * w, w * eha.epistemic_var(p, share), T, open_)
        i, a = np.nonzero(open_ & (val > EPS))
        unit.append(rows["_unit"].to_numpy()[i])
        action.append(a)
        value.append(val[i, a])
        top.append(best[i, a])
        do_by.append(day.toordinal() - lead_of[a])
    empty = np.empty(0, dtype=np.int64)
    return {
        "unit": np.concatenate(unit) if unit else empty,
        "action": np.concatenate(action) if action else empty,
        "value": np.concatenate(value) if value else np.empty(0, dtype=float),
        "top": np.concatenate(top) if top else empty,
        "do_by": np.concatenate(do_by) if do_by else empty,
    }


def _values(risk: np.ndarray, info: np.ndarray, T: np.ndarray,
            open_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EHA per (veteran, action), fixed before anything is chosen, and the need it helps most.

    `risk` is weighted risk w*p (n x K); `info` is weighted epistemic variance (n x K), the
    check-in's value, which sits in the last column. A veteran's prevention actions are taken
    in descending standalone EHA and each is valued on the risk left after all of the ones
    above it, whether or not those are chosen -- so the value never depends on capacity, and
    no subset of a veteran's actions can sum to more than the harm they carry that day.

    An action the veteran is not eligible for is worth nothing and takes nothing off the
    risk the next one works on, which is what zeroing its row of T does.
    """
    a_n = len(T)
    gain = np.einsum("nk,ak->nak", risk, T)             # per-need EHA of every action
    gain = np.concatenate([gain, info[:, None, :]], axis=1)
    gain[~open_] = 0.0

    standalone = gain[:, :a_n].sum(axis=2)              # n x A
    order = np.argsort(-standalone, axis=1, kind="stable")   # ties fall to the action index
    live = np.take_along_axis(standalone, order, axis=1) > EPS
    t_ordered = np.where(live[:, :, None], T[order], 0.0)
    left = np.empty_like(t_ordered)
    left[:, 0] = 1.0
    np.cumprod(1 - t_ordered[:, :-1], axis=1, out=left[:, 1:])
    np.put_along_axis(gain[:, :a_n], order[:, :, None],
                      risk[:, None, :] * left * t_ordered, axis=1)
    return gain.sum(axis=2), gain.argmax(axis=2)


# --------------------------------------------------------------------------- #
# One do-by day: the greedy pass, and the baselines that repeat it
# --------------------------------------------------------------------------- #

def _split_by_day(cand: dict[str, np.ndarray]):
    """The candidates, grouped by do-by day, oldest first."""
    order = np.argsort(cand["do_by"], kind="stable")
    ordered = {k: v[order] for k, v in cand.items()}
    cuts = np.flatnonzero(np.diff(ordered["do_by"])) + 1
    for part in np.split(np.arange(len(order)), cuts):
        if part.size:
            yield {k: v[part] for k, v in ordered.items()}


def _rounds(who: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Which action this is for its veteran, 0 for their best: the round it is offered in.

    Ranked by descending EHA, the same order the veteran's own actions were valued in, so
    `round` rising and `value` falling never disagree inside one veteran. That is exactly
    what the monotonicity argument needs, and it is why the round may sit above the band in
    the sort without putting the capacity slider's story at risk.

    Two sort keys, not three: `lexsort` is stable, so a veteran's exactly-equal actions keep
    the order `_candidates` emitted them in, which is already deterministic. The third key
    the greedy uses to break those ties costs a third of this function and decides nothing.
    """
    order = np.lexsort((-value, who))
    grouped = who[order]
    starts = np.flatnonzero(np.r_[True, grouped[1:] != grouped[:-1]])
    within = np.arange(len(order)) - np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    out = np.empty(len(order), dtype=np.int64)
    out[order] = within
    return out


def _allocate_do_by_day(day: dict[str, np.ndarray], cap: dict[str, int],
                        floor: dict[str, float], act_names: list[str], slots: np.ndarray,
                        tier_band: np.ndarray, vet: np.ndarray, groups: dict[str, list],
                        orders: dict[str, np.ndarray],
                        n_vets: int) -> tuple[dict[str, np.ndarray] | None, dict[str, float]]:
    """Who gets what on one work day, and what each baseline would have averted instead."""
    unit, a_idx, value = day["unit"], day["action"], day["value"]
    bucket = [ACTION_COST_UNIT[a] for a in act_names]
    who = vet[unit]

    # Two limits and two priorities, all fixed before anything is chosen, all inside this
    # one day. The limits are per risk day, because the tier is a statement about that day's
    # risk -- a Self-serve Friday earns one action about Friday, whoever else the veteran is
    # the rest of the week -- and per work day, because that is how often anyone may be
    # contacted in one morning: the most any of the risk days being worked for them today
    # allows. The band is the same rule read the same way round: the *best* tier among the
    # risk days being worked for them today, so a veteran whose Wednesday is Act-now is
    # worked as Act-now on the Monday their booking is due.
    per_day = np.zeros(n_vets, dtype=np.int64)
    np.maximum.at(per_day, who, slots[unit])
    band = np.full(n_vets, max(tiers.BAND.values()), dtype=np.int64)
    np.minimum.at(band, who, tier_band[unit])
    limit, cost = per_day.tolist(), slots.tolist()
    # Round and band as one integer, because lexsort pays per key and this runs three times
    # per work day on every slider drag. Round is the major digit; band never reaches N_BANDS.
    priority = _rounds(who, value) * len(tiers.BAND) + band[who]

    def pick(order_by: np.ndarray | None) -> np.ndarray:
        """One greedy pass: everyone's first action before anyone's second, Act-now band
        first inside each round, highest value inside that -- or, for a baseline, veterans
        from the highest `order_by` down in place of the value. (value, veteran, action)
        breaks ties the same way every run.

        Keying on the *veteran's* band and the action's round, rather than on the candidate's
        own tier, is what keeps more capacity from averting less harm; the module docstring
        has the argument.

        The loop body reads plain Python lists rather than indexing the numpy arrays. It
        runs once per candidate per baseline, four times over on every slider drag, and
        numpy scalar indexing inside it costs several times what the greedy itself does.
        """
        if order_by is None:
            order = np.lexsort((a_idx, who, -value, priority))
        else:
            order = np.lexsort((a_idx, -value, who, -order_by[unit], priority))
        seq = list(zip(order.tolist(), who[order].tolist(), unit[order].tolist(),
                       [bucket[a] for a in a_idx[order].tolist()], strict=True))

        used: dict[int, int] = {}
        used_risk: dict[int, int] = {}
        remaining = dict(cap)
        quota = {(col, g, b): math.floor(f * cap[b] + 1e-9)
                 for col, f in floor.items() for g in set(groups[col]) for b in cap}
        taken = [False] * len(seq)

        def run(reserved_only: bool) -> None:
            for c, i, u, b in seq:
                if (taken[c] or used.get(i, 0) >= limit[i] or remaining[b] <= 0
                        or used_risk.get(u, 0) >= cost[u]):
                    continue
                mine = [(col, groups[col][u], b) for col in floor]
                if reserved_only and not any(quota[q] > 0 for q in mine):
                    continue
                taken[c] = True
                used[i] = used.get(i, 0) + 1
                used_risk[u] = used_risk.get(u, 0) + 1
                remaining[b] -= 1
                for q in mine:
                    quota[q] = max(0, quota[q] - 1)

        if floor:
            run(reserved_only=True)
        run(reserved_only=False)
        return np.array(taken, dtype=bool)

    totals = {name: float(value[pick(col)].sum()) for name, col in orders.items()}
    mine = pick(None)
    if not mine.any():
        return None, totals
    return {k: v[mine] for k, v in day.items()}, totals


# --------------------------------------------------------------------------- #
# The rows
# --------------------------------------------------------------------------- #

def _rows(chosen: list[dict[str, np.ndarray]], frame: pl.DataFrame, act_names: list[str],
          lead_of: np.ndarray,
          rules: dict[str, rule_table.Rule] | None = None) -> pl.DataFrame:
    """The chosen candidates as the `actions` table, ranked within each do-by day."""
    picked = {k: np.concatenate([c[k] for c in chosen]) for k in chosen[0]}
    meta = frame.select(
        "_unit", "veteran_id", "tier", "tier_rule",
        *[f"{f}_{k}" for f in _PER_NEED for k in NEEDS],
        service=pl.when("ckd_dialysis").then(pl.lit("dialysis"))
                  .when("active_cancer_tx").then(pl.lit("infusion"))
                  .otherwise(pl.lit("opioid treatment program")),
    )
    joined = pl.DataFrame({"_unit": picked["unit"].astype(np.uint32)}).join(
        meta, on="_unit", how="left")          # left join keeps the candidate order

    k_idx = picked["top"]
    fields = {f: joined.select([f"{f}_{k}" for k in NEEDS]).to_numpy() for f in _PER_NEED[:4]}
    drivers = joined.select([f"driver_1_{k}" for k in NEEDS]).rows()
    vids = joined["veteran_id"].to_list()
    tiers_ = joined["tier"].to_list()
    fired = joined["tier_rule"].to_list()
    services = joined["service"].to_list()
    dates = [Date.fromordinal(int(d)) for d in picked["do_by"]]
    # A rule-triggered row carries its own mechanism, in the words a clinician signed off.
    because = {name: rule.because for name, rule in (rules or {}).items()}

    out = []
    for n in range(len(vids)):
        act, need, k = act_names[picked["action"][n]], NEEDS[k_idx[n]], int(k_idx[n])
        on, vid, tier = dates[n], vids[n], tiers_[n]
        aid = hashlib.sha1(f"{on}|{vid}|{act}".encode()).hexdigest()[:12]
        p = fields["p_mean"][n, k]
        out.append({
            "action_id": aid, "date": on, "veteran_id": vid, "action": act, "tier": tier,
            "eha": float(picked["value"][n]), "rank": 0,
            "lead_days": int(lead_of[picked["action"][n]]),
            "capacity_bucket": ACTION_COST_UNIT[act],
            "rationale": _rationale(act, need, p, fields["p_lo80"][n, k],
                                    fields["p_hi80"][n, k],
                                    fields["p_epistemic_share"][n, k], drivers[n][k],
                                    services[n], because.get(fired[n])),
            "headline": _headline(need, p, tier),
            "owner": OWNER.get(act, "care_team"), "message_id": f"msg-{aid}",
            "top_need": need, "top_driver": drivers[n][k],
        })
    return (pl.DataFrame(out, schema=OUT_SCHEMA)
              .sort(["date", "eha", "veteran_id", "action"],
                    descending=[False, True, False, False])
              .with_columns(rank=pl.int_range(1, pl.len() + 1, dtype=pl.Int32).over("date")))


def _headline(need: str, p: float, tier: str) -> str:
    """The one line a wall-mounted board may show beside a de-identified handle.

    Urgency and timing, and nothing a passer-by could turn back into a chart: no condition,
    no medicine, no service. "A gap in treatment, 34% chance -- act today".

    `_rationale` below is the other half of the same row and keeps all three, because the
    care-team drill-down is a private screen and cannot be acted on without them. Splitting
    them is the point: weakening `rationale` to fit the board would leave the person who
    picks up the phone with a sentence they cannot use.
    """
    return f"{NEED_HEADLINE[need]}, {p:.0%} chance — {TIER_URGENCY[tier]}"


def _rationale(action: str, need: str, p: float, lo: float, hi: float, share: float,
               driver: str | None, service: str, because: str | None = None) -> str:
    """One sentence a care-team member can read aloud. Names the service, the medicine and
    the driver on purpose -- so it is a drill-down line, never a board line (`_headline`).

    `because` is the `act_now.yaml` row that made this veteran Act-now, if one did. It goes
    last and in full: the probability did not trigger the rule, so a care team told to act
    today needs the mechanism that did, in the words a clinician signed off on.
    """
    if action == CHECK_IN:
        said = (f"Three-minute check-in to find out: {p:.0%} chance of {NEED_PHRASE[need]}, "
                f"and {share:.0%} of the uncertainty is what we do not know about them.")
    else:
        why = f", driven by {driver}" if driver else ""
        said = (f"{LEAD[action].format(service=service)}: {p:.0%} chance of "
                f"{NEED_PHRASE[need]} (80% interval {lo:.0%}-{hi:.0%}){why}.")
    return f"{said} {because}" if because else said


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=OUT_SCHEMA)


# --------------------------------------------------------------------------- #
# Input checks
# --------------------------------------------------------------------------- #

def _check_capacity(capacity: dict[str, int]) -> dict[str, int]:
    unknown = sorted(set(capacity) - set(BUCKETS))
    if unknown:
        raise ValueError(f"unknown capacity buckets {unknown}; buckets are {BUCKETS}")
    out = {b: int(capacity.get(b, 0)) for b in BUCKETS}
    negative = {b: v for b, v in out.items() if v < 0}
    if negative:
        raise ValueError(f"capacity cannot be negative: {negative}")
    return out


def _check_lead(lead: Mapping[str, int], act_names: list[str]) -> np.ndarray:
    """Lead days per action, in `act_names` order. Every proposable action needs one."""
    missing = [a for a in act_names if a not in lead]
    if missing:
        raise ValueError(f"no lead time for {missing}; every action allocate() can propose "
                         "needs one in tau.yaml")
    bad = {a: lead[a] for a in act_names if not isinstance(lead[a], int) or lead[a] < 0}
    if bad:
        raise ValueError(f"lead days must be whole days, zero or more: {bad}")
    return np.array([lead[a] for a in act_names], dtype=np.int64)


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
                    help="allocate one do-by day (default: every day the window reaches)")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    # hazards and site_status are not optional here: without them no Act-now rule can fire
    # and this would quietly write an actions.parquet missing the four rows the demo is made
    # of. `make hazards` produces both, and `make score` runs after it.
    actions, _, too_late = compare(schema.read("scores"), schema.read("cohort"),
                                   DEFAULT_CAPACITY, date=args.date,
                                   hazards=schema.read("hazards"),
                                   site_status=schema.read("site_status"))
    path = schema.write(actions, "actions")
    print(f"  actions  {actions.height:,} rows over {actions['date'].n_unique()} do-by days, "
          f"total EHA {total_eha(actions):,.1f}, {time.perf_counter() - t0:.1f}s -> {path}")
    if too_late:
        print(f"           {too_late:,} actions were already too late to book when the "
              f"forecast arrived")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

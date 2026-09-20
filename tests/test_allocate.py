"""Acceptance tests for the decision layer (SPEC §7): severity, tau, EHA, tiers, allocate.

The five-veteran case is small enough to check by hand against the SPEC §7.1 and §7.2
tables, and small enough to brute-force, so it is checked both ways. Everything else runs
against the panel the suite builds for itself (`tests/tables.py`) and seeded random
instances -- never against `data/`, whose contents depend on which make target ran last.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from datetime import date

import numpy as np
import polars as pl
import pytest

from leeward import schema
from leeward.decision import eha, severity, tau, tiers
from leeward.decision.allocate import allocate, compare, total_eha
from leeward.schema import ACTION_COST_UNIT, ACTIONS, DEFAULT_CAPACITY, NEEDS
from tables import table

DAY = date(2026, 7, 16)

# SPEC §7.1 and §7.2, retyped on purpose: the YAML must match the SPEC, not itself.
SPEC_W = {"breathing": 3, "heat": 4, "mental": 4, "treatment_gap": 5, "access_loss": 3}
SPEC_TAU = {                       # breathing, heat, mental, treatment_gap, access_loss
    "care_team_call":              (0.25, 0.30, 0.35, 0.40, 0.20),
    "early_refill":                (0.15, 0.10, 0.05, 0.60, 0.00),
    "cooling_center_ride":         (0.05, 0.60, 0.05, 0.00, 0.10),
    "clean_air_room":              (0.50, 0.05, 0.00, 0.00, 0.00),
    "backup_power_plan":           (0.20, 0.00, 0.05, 0.50, 0.10),
    "alt_site_booking":            (0.00, 0.00, 0.05, 0.70, 0.30),
    "evacuation_assist":           (0.05, 0.10, 0.10, 0.30, 0.75),
    "verified_text":               (0.05, 0.15, 0.10, 0.10, 0.05),
    "switch_to_local_pickup":      (0.10, 0.05, 0.05, 0.65, 0.10),
    "pharmacist_med_review":       (0.15, 0.45, 0.10, 0.20, 0.00),
    "cold_chain_plan":             (0.05, 0.10, 0.00, 0.55, 0.05),
    "controlled_substance_bridge": (0.00, 0.00, 0.20, 0.70, 0.20),
}

# A veteran with nothing unusual: no meds, no equipment, not site-dependent, not in a zone.
PLAIN = dict(
    borough="Queens", n_active_meds=0, mail_order_pharmacy=False, powered_equipment="none",
    ckd_dialysis=False, active_cancer_tx=False, on_methadone_otp=False, evac_zone=0,
    med_thermoreg_score=0.0, acb_score=0, med_combo_raas_diuretic=False,
    med_renal_triple=False, med_narrow_ti=False, med_cold_chain=False, med_controlled=False,
    caregiver="informal_coresident", low_assets=False,
)


def _cohort(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame([{**PLAIN, **r} for r in rows])


def _scores(risk: dict[str, dict[str, tuple[float, float]]], day: date = DAY) -> pl.DataFrame:
    """risk[veteran][need] = (p_mean, epistemic_share); unnamed needs are (0, 0)."""
    rows = []
    for vid, needs in risk.items():
        for k in NEEDS:
            p, s = needs.get(k, (0.0, 0.0))
            rows.append(dict(veteran_id=vid, date=day, need=k, p_mean=p,
                             p_lo80=round(p * 0.7, 4), p_hi80=round(min(1.0, p * 1.3), 4),
                             p_epistemic_share=s, driver_1=f"{k} driver"))
    return pl.DataFrame(rows)


def _cap(**kw: int) -> dict[str, int]:
    return {b: kw.get(b, 0) for b in DEFAULT_CAPACITY}


def _risk_day(actions: pl.DataFrame) -> pl.DataFrame:
    """`actions` with the day the risk lands beside the day the work is done."""
    return actions.with_columns(
        risk_date=pl.col("date") + pl.duration(days=pl.col("lead_days")))


#: Every lead time at zero, which is what this file's hand-checked cases run under. The
#: five-veteran world is the *allocator's* proof -- greedy order, marginal values, buckets,
#: the group floor -- and one day is enough to hold it. Lead time moves an action to another
#: day and is checked where it belongs, in tests/test_schedule.py, against a window wide
#: enough for a five-day lead to land in.
NO_LEAD = dict.fromkeys(tau.leads(), 0)


# --------------------------------------------------------------------------- #
# The editable tables
# --------------------------------------------------------------------------- #

def test_weights_and_tau_match_the_spec_tables() -> None:
    assert severity.load() == {k: float(v) for k, v in SPEC_W.items()}
    loaded = tau.load()
    assert set(loaded) == set(SPEC_TAU), "tau.yaml rows must be the SPEC §7.2 prevention actions"
    for a, row in SPEC_TAU.items():
        assert tuple(loaded[a][k] for k in NEEDS) == row, f"tau.yaml disagrees with SPEC for {a}"


def test_every_action_has_an_eligibility_rule() -> None:
    assert set(eha.ELIGIBLE) == set(ACTIONS)


@pytest.mark.parametrize("bad", [
    "breathing: 3\nheat: 4\nmental: 4\ntreatment_gap: 5\n",                    # need missing
    "breathing: 3\nheat: 4\nmental: 4\ntreatment_gap: 5\naccess_loss: -1\n",   # not positive
    "breathing: 3\nheat: 4\nmental: 4\ntreatment_gap: 5\naccess_loss: 3\nsmoke: 2\n",
])
def test_severity_typos_fail_loudly(tmp_path, bad: str) -> None:
    f = tmp_path / "severity.yaml"
    f.write_text(bad)
    with pytest.raises(ValueError):
        severity.load(f)


@pytest.mark.parametrize("bad", [
    "care_team_call: {breathing: 1.5, heat: 0, mental: 0, treatment_gap: 0, access_loss: 0}",
    "phone_tree: {breathing: 0.1, heat: 0, mental: 0, treatment_gap: 0, access_loss: 0}",
    "check_in_call: {breathing: 0.1, heat: 0, mental: 0, treatment_gap: 0, access_loss: 0}",
    "care_team_call: {breathing: 0.1, heat: 0, mental: 0, treatment_gap: 0}",
])
def test_tau_typos_fail_loudly(tmp_path, bad: str) -> None:
    f = tmp_path / "tau.yaml"
    f.write_text(bad)
    with pytest.raises(ValueError):
        tau.load(f)


def test_epistemic_variance_is_recovered_exactly() -> None:
    """Var + E[p(1-p)] = p(1-p) over draws, so the stored share pins the variance down."""
    draws = np.random.default_rng(7).beta(3, 12, 4000)
    p, var = draws.mean(), draws.var()
    share = var / (var + (draws * (1 - draws)).mean())
    assert eha.epistemic_var(p, share) == pytest.approx(var, rel=1e-9)


# --------------------------------------------------------------------------- #
# Five veterans, checked by hand
# --------------------------------------------------------------------------- #
#
# Weighted risk w*p, from SPEC_W:
#   V1  dialysis, treatment_gap p=.40, share .1 -> act_now (w=5, p>=.25, share<.4); 5*.40 = 2.0
#   V2  heat p=.30, share .1                    -> act_now (w=4);                   4*.30 = 1.2
#   V3  mental p=.20, share .5                  -> find_out (share>=.4, p>=.10);    4*.20 = 0.8
#   V4  breathing p=.10, share .1               -> self_serve (w=3 < 4);            3*.10 = 0.3
#   V5  every need p=.01                        -> everyday, so no action at all
#
# Capacity: call 2, booking 1, ride 1, free 10, everything else 0.
#
# Every value is fixed before anything is chosen. Each veteran's prevention actions are taken
# in descending standalone EHA and valued on the risk the ones above them leave behind -- V3
# and V4 included, even though one slot means they can only take the first of theirs here,
# because the schedule can spread a veteran's actions over several days (tests/test_schedule.py)
# and standalone values would then add up to more harm than the veteran carries:
#   V1  alt_site 2.0*.70 = 1.40   call (2.0*.30)*.40 = .24   text (.60*.60)*.10 = .036
#   V2  ride 1.2*.60 = .72   call (1.2*.40)*.30 = .144   text (.48*.70)*.15 = .0504
#   V3  check_in VOI = 4 * .5*.2*.8 = .32  (w * share * p(1-p))   call .8*.35 = .28
#   V4  clean_air .3*.50 = .15   call (.3*.50)*.25 = .0375   ride (.1125)*.05 = .005625
#       text (.106875)*.05 = .00534375
#
# Greedy, highest first:
#   1.40   V1 alt_site   -> booking full
#    .72   V2 ride       -> ride full
#    .32   V3 check_in   -> call 1 of 2. V3 has its one action, so V3's call (.28) is skipped
#    .24   V1 call       -> call full. V4 clean_air .15, V2 call .144, V4 call .0375: no room
#    .0504 V2 text,  .036 V1 text (V1's third),  .00534375 V4 text
#
# Reported total 2.77174375. The chosen set truly averts 2.803, because V2 never got the call
# and V4 never got the clean-air room, so their texts really avert .48*.15 = .072 and
# .3*.05 = .015. Reported EHA is a floor, never a ceiling.

FIVE_RISK = {
    "V1": {"treatment_gap": (0.40, 0.1)},
    "V2": {"heat": (0.30, 0.1)},
    "V3": {"mental": (0.20, 0.5)},
    "V4": {"breathing": (0.10, 0.1)},
    "V5": {k: (0.01, 0.1) for k in NEEDS},
}
FIVE_COHORT = [
    {"veteran_id": "V1", "ckd_dialysis": True},
    {"veteran_id": "V2"}, {"veteran_id": "V3"}, {"veteran_id": "V4"}, {"veteran_id": "V5"},
]
FIVE_CAP = dict(call=2, booking=1, ride=1, free=10)
FIVE_EXPECTED = [                  # rank order
    ("V1", "alt_site_booking", 1.40, "act_now"),
    ("V2", "cooling_center_ride", 0.72, "act_now"),
    ("V3", "check_in_call", 0.32, "find_out"),
    ("V1", "care_team_call", 0.24, "act_now"),
    ("V2", "verified_text", 0.0504, "act_now"),
    ("V1", "verified_text", 0.036, "act_now"),
    ("V4", "verified_text", 0.00534375, "self_serve"),
]
_ANYONE = ["care_team_call", "verified_text", "cooling_center_ride", "clean_air_room"]
FIVE_MENU = {  # eligible actions and slots, by hand; V5 is Everyday and gets nothing
    "V1": (_ANYONE + ["alt_site_booking"], 3),
    "V2": (_ANYONE, 3),
    "V3": (_ANYONE + ["check_in_call"], 1),
    "V4": (_ANYONE, 1),
}


def _five(**cap: int) -> pl.DataFrame:
    return allocate(_scores(FIVE_RISK), _cohort(FIVE_COHORT), _cap(**{**FIVE_CAP, **cap}),
                    lead=NO_LEAD)


def test_five_veterans_hand_checked() -> None:
    got = _five()
    schema.validate(got, "actions")
    assert got.sort("rank").select("veteran_id", "action", "tier").rows() == [
        (v, a, t) for v, a, _, t in FIVE_EXPECTED]
    assert got.sort("rank")["eha"].to_list() == pytest.approx([e for _, _, e, _ in FIVE_EXPECTED])
    assert got["rank"].to_list() == list(range(1, 8))
    assert total_eha(got) == pytest.approx(2.77174375)
    assert "V5" not in got["veteran_id"].to_list(), "an Everyday veteran got an action"


def _true_value(v: str, acts: tuple[str, ...]) -> float:
    """Harm a set of actions really averts for one of the five, computed without allocate.py."""
    w = np.array([SPEC_W[k] for k in NEEDS])
    risk = np.array([FIVE_RISK[v].get(k, (0, 0))[0] for k in NEEDS])
    share = np.array([FIVE_RISK[v].get(k, (0, 0))[1] for k in NEEDS])
    left = np.ones(len(NEEDS))
    for a in acts:
        if a != "check_in_call":
            left *= 1 - np.array(SPEC_TAU[a])
    averted = float((w * risk * (1 - left)).sum())
    if "check_in_call" in acts:
        averted += float((w * share * risk * (1 - risk)).sum())
    return averted


def _brute_force_optimum() -> float:
    """The best feasible assignment of the five, by trying every one."""
    options = {v: [c for n in range(lim + 1) for c in itertools.combinations(acts, n)]
               for v, (acts, lim) in FIVE_MENU.items()}
    best = 0.0
    for combo in itertools.product(*options.values()):
        used = Counter(ACTION_COST_UNIT[a] for acts in combo for a in acts)
        if all(used[b] <= FIVE_CAP.get(b, 0) for b in used):
            best = max(best, sum(_true_value(v, a) for v, a in zip(options, combo, strict=True)))
    return best


def test_five_veterans_choice_is_the_brute_force_optimum() -> None:
    got = _five()
    chosen = {v: tuple(got.filter(pl.col("veteran_id") == v)["action"]) for v in FIVE_MENU}
    truly = sum(_true_value(v, acts) for v, acts in chosen.items())
    assert truly == pytest.approx(_brute_force_optimum())
    assert truly == pytest.approx(2.803)
    assert total_eha(got) <= truly + 1e-12, "reported EHA must never exceed what is averted"


@pytest.mark.parametrize("bucket", ["call", "booking", "ride", "free"])
def test_five_veterans_more_of_any_bucket_never_averts_less(bucket: str) -> None:
    totals = [total_eha(_five(**{bucket: n})) for n in range(6)]
    assert totals == sorted(totals), f"{bucket}: {totals}"


# --------------------------------------------------------------------------- #
# Baselines: the same team and the same candidates, worked in a different order
# --------------------------------------------------------------------------- #
#
# Same five veterans, same capacity, same values. A baseline visits veterans in its own order;
# each takes their best actions that still have room. Worked by hand:
#
#   oldest first  V4 (99), V3 (88), V2 (77), V1 (66)
#     V4  clean_air .15 takes the only ride slot
#     V3  check_in .32 takes call 1 of 2
#     V2  ride is gone; call .144 takes call 2 of 2; text .0504
#     V1  alt_site 1.40 takes the booking; call is gone; text .036
#     total .15 + .32 + .1944 + 1.436 = 2.1004
#
#   in the allocator's own veteran order  V1, V2, V3, V4
#     V1  alt_site 1.40, call .24, text .036     V2  ride .72, call .144, text .0504
#     V3  check_in and call are gone; text .052  V4  ride and call are gone; text .00534375
#     total 1.676 + .9144 + .052 + .00534375 = 2.64774375
#
# The allocator's own 2.77174375 is higher because it gave V3 the check-in (.32) ahead of V2's
# and V1's later calls; ranking by anything else cannot see that.

FIVE_AGES = {"V1": 66, "V2": 77, "V3": 88, "V4": 99, "V5": 55}


def _five_compare(rank_by: dict[str, str], ages: dict[str, int] = FIVE_AGES, **cap: int):
    cohort = _cohort([{**c, "age": ages[c["veteran_id"]], "rank_key": 0}
                      for c in FIVE_COHORT])
    return compare(_scores(FIVE_RISK), cohort, _cap(**{**FIVE_CAP, **cap}), rank_by=rank_by,
                   lead=NO_LEAD)


def test_baselines_hand_checked() -> None:
    got, totals, _, _ = _five_compare({"rank_by_age": "age", "in_order": "rank_key"})
    assert total_eha(got) == pytest.approx(2.77174375), "the allocator's own list must not change"
    assert totals["rank_by_age"] == pytest.approx(2.1004)
    assert totals["in_order"] == pytest.approx(2.64774375), "all keys tied means veteran order"


def test_compare_returns_exactly_what_allocate_returns() -> None:
    got, _, _, _ = _five_compare({"rank_by_age": "age"})
    assert got.equals(_five())


def test_a_baseline_with_no_names_is_just_allocate() -> None:
    got, totals, _, _ = compare(_scores(FIVE_RISK), _cohort(FIVE_COHORT), _cap(**FIVE_CAP),
                             lead=NO_LEAD)
    assert totals == {} and got.equals(_five())


def test_a_baseline_on_a_missing_column_fails_loudly() -> None:
    with pytest.raises(ValueError, match="rank_by"):
        compare(_scores(FIVE_RISK), _cohort(FIVE_COHORT), _cap(**FIVE_CAP),
                rank_by={"by_shoe_size": "shoe_size"}, lead=NO_LEAD)


def test_a_baseline_with_nothing_to_score_totals_zero() -> None:
    cohort = _cohort([{**c, "age": 70} for c in FIVE_COHORT])
    _, totals, _, _ = compare(_scores(FIVE_RISK), cohort, _cap(**FIVE_CAP),
                           rank_by={"x": "age"}, date=date(2000, 1, 1), lead=NO_LEAD)
    assert totals == {"x": 0.0}


# --------------------------------------------------------------------------- #
# Seeded random instances: the monotonicity property, bucket by bucket
# --------------------------------------------------------------------------- #

BOROUGHS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

#: Seeds on which re-valuing Act-now actions as they were chosen let one more unit of
#: capacity lower the total (539: +1 evac, 11.55 -> 10.79). Pinned so that design stays gone.
BROKE_LAZY_GREEDY = [183, 539, 1632, 2319, 2526]


def _random_instance(seed: int):
    """Small, Act-now-heavy instances: the shape that breaks a careless allocator."""
    r = np.random.default_rng(seed)
    n = int(r.integers(2, 9))
    vids = [f"R{i}" for i in range(n)]
    cohort = _cohort([{
        "veteran_id": v, "borough": BOROUGHS[int(r.integers(0, 5))],
        "n_active_meds": int(r.integers(0, 3)), "mail_order_pharmacy": bool(r.random() < .8),
        "powered_equipment": "oxygen" if r.random() < .4 else "none",
        "ckd_dialysis": bool(r.random() < .4), "evac_zone": int(r.integers(0, 2)),
        "med_thermoreg_score": float(r.random() < .5), "med_cold_chain": bool(r.random() < .4),
        "med_controlled": bool(r.random() < .4)} for v in vids])
    act_heavy = r.random() < 0.7
    risk = {v: {k: (float(r.beta(2, 3) if act_heavy else r.beta(1.2, 4)),
                    float(r.uniform(0, 0.35) if act_heavy else r.uniform(0, 0.8)))
                for k in NEEDS} for v in vids}
    cap = {b: int(r.integers(0, 3)) for b in sorted(set(ACTION_COST_UNIT.values()))}
    floor = {"borough": 0.2} if r.random() < 0.3 else None
    return _scores(risk), cohort, cap, floor


@pytest.mark.parametrize("seed", BROKE_LAZY_GREEDY + list(range(40)))
def test_random_instances_more_of_any_bucket_never_averts_less(seed: int) -> None:
    scores, cohort, cap, floor = _random_instance(seed)
    base = total_eha(allocate(scores, cohort, cap, floor, lead=NO_LEAD))
    for bucket in sorted(set(ACTION_COST_UNIT.values())):
        more = total_eha(allocate(scores, cohort, dict(cap, **{bucket: cap[bucket] + 1}), floor,
                                  lead=NO_LEAD))
        assert more >= base - 1e-9, f"seed {seed}: +1 {bucket} lowered EHA {base} -> {more}"


# --------------------------------------------------------------------------- #
# Against the full panel the suite builds for itself (tests/tables.py)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def fixtures() -> tuple[pl.DataFrame, pl.DataFrame]:
    return table("scores"), table("cohort")


@pytest.fixture(scope="module")
def default_actions(fixtures) -> pl.DataFrame:
    scores, cohort = fixtures
    return allocate(scores, cohort, DEFAULT_CAPACITY)


def test_panel_actions_match_the_contract(default_actions) -> None:
    schema.validate(default_actions, "actions")


def test_a_risk_day_earns_a_list_exactly_when_someone_is_above_the_everyday_floor(
        fixtures, default_actions) -> None:
    """Not "every scored day gets a list" -- a day can correctly produce nothing.

    On the real 120-day Sandy run, 14 days do: max heat 81.9F under the 82F threshold, no
    smoke, no outage, no flood, peak risk 0.0476 under the 0.05 self-serve floor. The whole
    panel is Everyday tier, and a care team doing nothing on a calm June day is the right
    answer. What must hold is the biconditional, not the count.

    The biconditional is about the day the *risk* lands, which is `date + lead_days`; which
    day the work falls on is the schedule's business (tests/test_schedule.py).
    """
    scores, _ = fixtures
    graded = tiers.assign(scores)
    actionable = set(graded.filter(pl.col("tier") != "everyday")["date"].to_list())
    assert set(_risk_day(default_actions)["risk_date"].to_list()) == actionable


def test_a_day_where_the_whole_panel_is_everyday_produces_no_actions() -> None:
    """That calm day, pinned on purpose instead of left to whatever the panel happens to hold."""
    vets = ["V1", "V2", "V3"]
    calm = _scores({v: {k: (0.001, 0.1) for k in NEEDS} for v in vets})
    cohort = _cohort([{"veteran_id": v} for v in vets])
    assert (tiers.assign(calm)["tier"] == "everyday").all(), "this day was meant to be calm"
    got = allocate(calm, cohort, DEFAULT_CAPACITY)
    schema.validate(got, "actions")            # an empty list is still a legal actions table
    assert got.height == 0, "a panel with nobody above the floor was given something to do"
    assert total_eha(got) == 0.0


def test_a_calm_day_earns_nothing_of_its_own_but_is_still_a_day_of_work() -> None:
    """A calm day's own panel buys it no actions -- and it still has a list, because the
    ride for tomorrow's heat has to be booked today."""
    calm, busy = date(2026, 6, 12), date(2026, 6, 13)
    scores = pl.concat([
        _scores({v: {k: (0.001, 0.1) for k in NEEDS} for v in ("V1", "V2")}, day=calm),
        _scores({"V1": {"heat": (0.30, 0.1)}, "V2": {"breathing": (0.10, 0.1)}}, day=busy)])
    cohort = _cohort([{"veteran_id": "V1"}, {"veteran_id": "V2"}])
    got = _risk_day(allocate(scores, cohort, DEFAULT_CAPACITY))
    assert got["risk_date"].unique().to_list() == [busy], "the calm day's own panel earns nothing"
    assert calm in got["date"].to_list(), "yet the calm day is when the ride gets booked"
    assert (got.filter(pl.col("date") == calm)["lead_days"] > 0).all()


def test_fixture_capacity_is_per_day_and_never_exceeded(default_actions) -> None:
    used = default_actions.group_by("date", "capacity_bucket").agg(pl.len().alias("n"))
    for _, bucket, n in used.iter_rows():
        assert n <= DEFAULT_CAPACITY[bucket], f"{n} in {bucket}, cap {DEFAULT_CAPACITY[bucket]}"
    calls = used.filter(pl.col("capacity_bucket") == "call")["n"]
    assert calls.max() == DEFAULT_CAPACITY["call"], "calls are the scarce bucket; it should fill"


def test_fixture_slot_limits_and_dense_ranks(default_actions) -> None:
    """Two slot limits, both inside one work day, because a work day carries work for
    several risk days and a risk day's work is spread over several days.

    Per (work day, risk day) the tier means what it always meant: a Self-serve Friday earns
    one action about Friday on any one morning. Per work day nobody is contacted more than
    three times, however many risk days are being worked for them.

    What is deliberately *not* capped is a risk day's actions summed over the mornings they
    are done on -- a refill five days out and a text on the day are two touches for one
    Friday, and both have to happen when they happen. That is safe because every one of a
    veteran's actions for a risk day is valued on the risk the others leave behind, so the
    harm they claim between them is still bounded by the harm the veteran carries
    (`test_fixture_harm_averted_never_exceeds_the_harm_there_is`).
    """
    per_risk = _risk_day(default_actions).group_by("date", "risk_date", "veteran_id").agg(
        pl.len().alias("n"), pl.col("tier").first(), pl.col("tier").n_unique().alias("tiers"))
    assert per_risk["tiers"].max() == 1, "a veteran has one tier per risk day"
    assert per_risk.filter((pl.col("tier") != "act_now") & (pl.col("n") > 1)).height == 0
    assert per_risk["n"].max() <= 3
    per_day = default_actions.group_by("date", "veteran_id").agg(pl.len().alias("n"))
    assert per_day["n"].max() <= 3, "no veteran is reached more than three times in a day"
    for _, g in default_actions.group_by("date"):
        g = g.sort("rank")
        assert g["rank"].to_list() == list(range(1, g.height + 1))
        assert g["eha"].to_list() == sorted(g["eha"].to_list(), reverse=True)
    assert default_actions["action_id"].n_unique() == default_actions.height


def test_fixture_actions_only_go_to_veterans_they_apply_to(fixtures, default_actions) -> None:
    scores, cohort = fixtures
    joined = default_actions.join(cohort, on="veteran_id")
    for action, rule in eha.ELIGIBLE.items():
        wrong = joined.filter((pl.col("action") == action) & ~rule)
        assert wrong.height == 0, f"{wrong.height} {action} rows for veterans it does not apply to"
    assert (joined.filter(pl.col("action") == "check_in_call")["tier"] == "find_out").all()
    everyday = (tiers.assign(scores).filter(pl.col("tier") == "everyday")
                .rename({"date": "risk_date"}))
    assert _risk_day(default_actions).join(
        everyday, on=["veteran_id", "risk_date"]).height == 0


def test_fixture_harm_averted_never_exceeds_the_harm_there_is(fixtures, default_actions) -> None:
    """Three actions on one heat illness cannot prevent 135% of it.

    Against the *risk* day, because that is where the harm is: two actions done on two
    different mornings for one Friday surge still share Friday's risk between them.
    """
    scores, _ = fixtures
    w = severity.load()
    harm = scores.group_by("veteran_id", "date").agg(
        (pl.col("p_mean") * pl.col("need").replace_strict(w)).sum().alias("harm")
    ).rename({"date": "risk_date"})
    averted = (_risk_day(default_actions).filter(pl.col("action") != "check_in_call")
               .group_by("veteran_id", "risk_date").agg(pl.col("eha").sum().alias("averted")))
    over = averted.join(harm, on=["veteran_id", "risk_date"]).filter(
        pl.col("averted") > pl.col("harm") + 1e-9)
    assert over.height == 0, f"{over.height} veteran-days avert more harm than they carry"


def test_medication_actions_go_to_the_pharmacist(default_actions) -> None:
    """Leeward never changes a medication; it flags the veteran for the pharmacist."""
    meds = default_actions.filter(pl.col("action").is_in(
        ["pharmacist_med_review", "early_refill", "switch_to_local_pickup",
         "controlled_substance_bridge"]))
    assert meds.height > 0
    assert (meds["owner"] == "pharmacist").all()
    assert meds["rationale"].str.contains("pharmacist").all()
    review = default_actions.filter(pl.col("action") == "pharmacist_med_review")
    assert review["rationale"].str.contains("decides").all()


def test_same_inputs_same_list(fixtures) -> None:
    scores, cohort = fixtures
    day = scores["date"].min()
    a = allocate(scores, cohort, DEFAULT_CAPACITY, date=day)
    b = allocate(scores, cohort, DEFAULT_CAPACITY, date=day)
    assert a.equals(b)


@pytest.mark.parametrize("bucket", ["call", "ride", "pharmacist_slot", "refill"])
def test_fixture_more_of_any_bucket_never_averts_less(fixtures, bucket: str) -> None:
    scores, cohort = fixtures
    day = scores["date"].min()
    totals = [total_eha(allocate(scores, cohort, dict(DEFAULT_CAPACITY, **{bucket: n}),
                                 date=day)) for n in (0, 5, 10, 20, 40, 80)]
    assert totals == sorted(totals), f"{bucket}: {totals}"


# --------------------------------------------------------------------------- #
# Group floor and input checks
# --------------------------------------------------------------------------- #

def test_group_floor_reserves_each_borough_its_share(fixtures) -> None:
    """A fifth of the calls reserved per borough: floor(0.2 x 20) = 4 each, five boroughs."""
    scores, cohort = fixtures
    day = scores["date"].min()
    share = 0.2
    cap = dict(DEFAULT_CAPACITY, call=20)
    reserved = math.floor(share * cap["call"])
    boroughs = sorted(cohort["borough"].unique().to_list())
    assert len(boroughs) == 5, f"the floor is sized for five boroughs, not {boroughs}"

    plain = allocate(scores, cohort, cap, date=day)
    floored = allocate(scores, cohort, cap, group_floor={"borough": share}, date=day)
    calls = floored.filter(pl.col("capacity_bucket") == "call").join(
        cohort.select("veteran_id", "borough"), on="veteran_id")
    by_boro = Counter(calls["borough"].to_list())
    for boro in boroughs:
        assert by_boro[boro] >= reserved, (
            f"{boro} got {by_boro[boro]} of {cap['call']} calls; floor is {reserved}")
    assert calls.height == cap["call"]
    assert total_eha(floored) <= total_eha(plain) + 1e-9, "a floor is a constraint; it costs EHA"


@pytest.mark.parametrize("floor", [{"borough": 0.3}, {"borough": -0.1}, {"zipcode": 0.1}])
def test_impossible_group_floor_is_refused(fixtures, floor) -> None:
    scores, cohort = fixtures
    with pytest.raises(ValueError):
        allocate(scores, cohort, DEFAULT_CAPACITY, group_floor=floor, date=scores["date"].min())


def test_capacity_is_checked() -> None:
    scores, cohort = _scores(FIVE_RISK), _cohort(FIVE_COHORT)
    with pytest.raises(ValueError):
        allocate(scores, cohort, {"phone": 3})
    with pytest.raises(ValueError):
        allocate(scores, cohort, _cap(call=-1))
    assert allocate(scores, cohort, {}).height == 0, "a bucket not named has no capacity"


def test_scores_for_unknown_veterans_are_refused() -> None:
    with pytest.raises(ValueError):
        allocate(_scores(FIVE_RISK), _cohort(FIVE_COHORT[:4]), _cap(**FIVE_CAP),
                 lead=NO_LEAD)

"""The four hazard-triggered Act-now rules (SPEC §7.5), and the tier priority band (§7.4).

Two things are under test here, and they are the same argument twice.

A probability is a summary of what the model has seen. The four rules encode mechanisms it
has *not* seen: a closed station, an excluded drug class, a stopped mail route, an outage in
a home with nobody in it. A veteran who matches one is Act-now whatever their p_mean says,
which is the whole point -- these are the rows a care team can act on with no argument.

Having said that, the allocator then has to believe it. Before the band, a Self-serve
veteran with broad moderate risk outranked an Act-now veteran with one sharp risk for the
same scarce call slot, so the tier badge promised something the list did not deliver. The
band is what makes the badge true.

Today's numbers, measured on this suite's panel at DEFAULT_CAPACITY before any of this
landed, are pinned in `BEFORE_*` below. Test (d) of the task is the 15% bar against them.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from leeward.decision import rules, severity, tau, tiers
from leeward.decision.allocate import allocate, compare, total_eha
from leeward.schema import DEFAULT_CAPACITY, NEEDS, TIERS
from tables import table

DAY = date(2026, 7, 16)
STATION = "630"          # Manhattan, evacuation zone 1, closed five months after Sandy
ZIP = "10010"

#: Every lead at zero, so a hand-built one-day case stays a one-day case. Lead time is
#: tested where it belongs, in tests/test_schedule.py.
NO_LEAD = dict.fromkeys(tau.leads(), 0)

#: Total EHA on this suite's panel at DEFAULT_CAPACITY *before* the tier band, recorded by
#: running `allocate(table("scores"), table("cohort"), DEFAULT_CAPACITY)` on the parent of
#: the commit that added it. Test (d) of the task: the band may not cost more than 15%.
#: Both numbers are pinned because the free bucket (verified_text) is unlimited and dilutes
#: the total -- the band can only ever be felt in the buckets that are actually scarce.
BEFORE_TOTAL_EHA = 5292.4522
BEFORE_SCARCE_EHA = 4931.0474
MAX_EHA_LOSS = 0.15

#: A veteran with nothing unusual about them: no meds, no equipment, not site-dependent.
PLAIN = dict(
    borough="Manhattan", modzcta=ZIP, facility_id=STATION, n_active_meds=0,
    mail_order_pharmacy=False, days_supply_remaining=90, powered_equipment="none",
    ckd_dialysis=False, active_cancer_tx=False, on_methadone_otp=False, evac_zone=0,
    med_thermoreg_score=0.0, acb_score=0, med_combo_raas_diuretic=False,
    med_renal_triple=False, med_narrow_ti=False, med_cold_chain=False, med_controlled=False,
    caregiver="informal_coresident", low_assets=False,
)

#: A quiet day: no outage, no disrupted mail, and the station open.
CALM_ZIP = dict(modzcta=ZIP, outage_frac=0.0, mail_delivery_disrupted=False)
CALM_STATION = dict(facility_id=STATION, site_down=False)


def _cohort(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame([{**PLAIN, **r} for r in rows])


def _hazards(rows: list[dict], day: date = DAY) -> pl.DataFrame:
    return pl.DataFrame([{**CALM_ZIP, "date": day, **r} for r in rows])


def _sites(rows: list[dict], day: date = DAY) -> pl.DataFrame:
    return pl.DataFrame([{**CALM_STATION, "date": day, **r} for r in rows])


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


def _assign(scores, cohort, hazards, sites):
    return tiers.assign(scores, cohort=cohort, hazards=hazards, site_status=sites)


def _tier(assigned: pl.DataFrame, vid: str) -> tuple[str, str | None]:
    row = assigned.filter(pl.col("veteran_id") == vid).row(0, named=True)
    return row["tier"], row["tier_rule"]


# --------------------------------------------------------------------------- #
# The rules file is the specification a clinician reads
# --------------------------------------------------------------------------- #

def test_the_rules_file_is_yaml_backed_beside_severity_and_tau() -> None:
    """Same promise as severity.yaml and tau.yaml: retuned without touching code."""
    assert rules.PATH.name == "act_now.yaml"
    assert rules.PATH.exists(), f"{rules.PATH} is missing"
    assert rules.PATH.parent == severity.PATH.parent == tau.PATH.parent, (
        "the rules table belongs beside severity.yaml and tau.yaml, where a clinician "
        "already knows to look")


def test_the_four_rules_are_all_there_and_each_one_cites_its_mechanism() -> None:
    loaded = rules.load()
    assert set(loaded) == set(rules.EXPECTED), (
        "act_now.yaml must hold exactly the four hazard-triggered rules of SPEC §7.5; "
        f"missing {sorted(set(rules.EXPECTED) - set(loaded))}, "
        f"unknown {sorted(set(loaded) - set(rules.EXPECTED))}")
    for name, rule in loaded.items():
        assert rule.because.strip(), f"{name} has no sentence a care team could read"
        assert rule.source.strip(), f"{name} cites no source"
        assert rule.when, f"{name} has no conditions, so it would fire on everyone"


@pytest.mark.parametrize("bad", [
    "{}",                                                   # no rules at all
    "not_a_rule:\n  veteran: ckd_dialysis\n  because: x\n  source: y\n",   # unknown rule
    "site_dependent_at_a_closed_station:\n  veteran: no_such_column\n"
    "  because: x\n  source: y\n",                          # column that does not exist
    "site_dependent_at_a_closed_station:\n  spleen: ckd_dialysis\n"
    "  because: x\n  source: y\n",                          # frame that does not exist
    "site_dependent_at_a_closed_station:\n  veteran: age\n"
    "  because: x\n  source: y\n",                          # bare non-boolean column
    "site_dependent_at_a_closed_station:\n  veteran: ckd_dialysis and age > 5 or ptsd\n"
    "  because: x\n  source: y\n",                          # 'and' and 'or' in one cell
    "site_dependent_at_a_closed_station:\n  veteran: caregiver <= none\n"
    "  because: x\n  source: y\n",                          # ordering a string column
    "site_dependent_at_a_closed_station:\n  veteran: ckd_dialysis\n  source: y\n",  # no because
])
def test_a_typo_in_the_rules_file_fails_loudly(tmp_path, bad: str) -> None:
    """Nobody reads diffs here, so a broken rule must not quietly stop firing."""
    path = tmp_path / "act_now.yaml"
    path.write_text(bad)
    with pytest.raises(ValueError):
        rules.load(path)


# --------------------------------------------------------------------------- #
# (a) and (b): the two rules the deck is built on
# --------------------------------------------------------------------------- #

def test_a_dialysis_patient_of_a_closed_station_is_act_now_under_the_line() -> None:
    """(a) The mechanism outranks the probability. p_mean 0.05 everywhere, still Act-now."""
    cohort = _cohort([{"veteran_id": "v-dialysis", "ckd_dialysis": True},
                      {"veteran_id": "v-plain"}])
    scores = _scores({"v-dialysis": {k: (0.05, 0.1) for k in NEEDS},
                      "v-plain": {k: (0.05, 0.1) for k in NEEDS}})

    open_day = _assign(scores, cohort, _hazards([{}]), _sites([{}]))
    assert _tier(open_day, "v-dialysis") == ("self_serve", None), (
        "with the station open, nothing here is above the Act-now line")

    closed = _assign(scores, cohort, _hazards([{}]), _sites([{"site_down": True}]))
    tier, rule = _tier(closed, "v-dialysis")
    assert tier == "act_now", (
        "a dialysis patient of a station that is down is Act-now whatever their p_mean: "
        "the model has not seen the closure, the rule has")
    assert rule == "site_dependent_at_a_closed_station"
    assert _tier(closed, "v-plain") == ("self_serve", None), (
        "the closure is not a reason to call someone whose care does not depend on the site")


def test_a_controlled_substance_at_a_closed_station_names_the_relief_plan() -> None:
    """(b) The rationale has to say *why* the usual fallback is not available to them."""
    cohort = _cohort([{"veteran_id": "v-controlled", "med_controlled": True,
                       "n_active_meds": 3}])
    scores = _scores({"v-controlled": {k: (0.05, 0.1) for k in NEEDS}})
    sites = _sites([{"site_down": True}])

    tier, rule = _tier(_assign(scores, cohort, _hazards([{}]), sites), "v-controlled")
    assert tier == "act_now"
    assert rule == "controlled_substance_at_a_closed_station"

    because = rules.load()[rule].because
    assert "Pharmacy Disaster Relief Plan" in because, (
        "the rule must cite the program by name: it is the reason a 10-day retail supply "
        "covers everyone else on this list and not this veteran")

    acted = allocate(scores, cohort, DEFAULT_CAPACITY, hazards=_hazards([{}]),
                     site_status=sites, lead=NO_LEAD)
    mine = acted.filter(pl.col("veteran_id") == "v-controlled")
    assert mine.height, "an Act-now veteran with capacity to spare must get something"
    assert any("Pharmacy Disaster Relief Plan" in r for r in mine["rationale"]), (
        "the reason the rule fired has to reach the person who picks up the phone, not "
        f"stop at the tier column; got {mine['rationale'].to_list()}")


# --------------------------------------------------------------------------- #
# The other two rules, and the hazard half of all four
# --------------------------------------------------------------------------- #

def test_mail_order_supply_short_on_a_disrupted_day_is_act_now() -> None:
    """~80% of VA outpatient prescriptions arrive by mail, so a flooded ZIP is a
    medication-supply event -- but only for the veterans whose supply runs out inside it."""
    cohort = _cohort([
        {"veteran_id": "v-short", "mail_order_pharmacy": True, "days_supply_remaining": 3,
         "n_active_meds": 2},
        {"veteran_id": "v-stocked", "mail_order_pharmacy": True, "days_supply_remaining": 60,
         "n_active_meds": 2},
        {"veteran_id": "v-pickup", "mail_order_pharmacy": False, "days_supply_remaining": 3,
         "n_active_meds": 2},
    ])
    scores = _scores({v: {k: (0.05, 0.1) for k in NEEDS}
                      for v in ("v-short", "v-stocked", "v-pickup")})

    quiet = _assign(scores, cohort, _hazards([{}]), _sites([{}]))
    assert _tier(quiet, "v-short")[0] == "self_serve", "no disruption, no rule"

    cut = _assign(scores, cohort, _hazards([{"mail_delivery_disrupted": True}]), _sites([{}]))
    assert _tier(cut, "v-short") == ("act_now", "mail_order_supply_short_on_a_disrupted_day")
    assert _tier(cut, "v-stocked")[0] == "self_serve", (
        "60 days of supply outlasts the forecast window; targeting by days-supply rather "
        "than mailing everyone is the point of the rule")
    assert _tier(cut, "v-pickup")[0] == "self_serve", (
        "a veteran who collects in person is not affected by the mail route")


def test_powered_equipment_with_no_caregiver_in_an_outage_is_act_now() -> None:
    """The emPOWER case: the equipment stops and nobody in the home can act on it."""
    cohort = _cohort([
        {"veteran_id": "v-alone", "powered_equipment": "oxygen_concentrator",
         "caregiver": "none"},
        {"veteran_id": "v-supported", "powered_equipment": "oxygen_concentrator",
         "caregiver": "informal_coresident"},
        {"veteran_id": "v-unpowered", "powered_equipment": "none", "caregiver": "none"},
    ])
    scores = _scores({v: {k: (0.05, 0.1) for k in NEEDS}
                      for v in ("v-alone", "v-supported", "v-unpowered")})

    dark = _assign(scores, cohort, _hazards([{"outage_frac": 0.4}]), _sites([{}]))
    assert _tier(dark, "v-alone") == ("act_now", "unattended_powered_equipment_in_an_outage")
    assert _tier(dark, "v-supported")[0] == "self_serve", (
        "a co-resident who can act on the equipment is exactly what the rule asks about")
    assert _tier(dark, "v-unpowered")[0] == "self_serve"

    lit = _assign(scores, cohort, _hazards([{"outage_frac": 0.0}]), _sites([{}]))
    assert _tier(lit, "v-alone")[0] == "self_serve", "no outage forecast, no rule"


def test_without_hazards_no_hazard_rule_can_fire() -> None:
    """`assign()` still answers from the scores alone, because other lanes call it that way.

    It is a quieter answer, not a wrong one: every rule here needs a hazard or a site
    status, and a caller who passes neither is asking what the probabilities alone say.
    """
    scores = _scores({"v-dialysis": {k: (0.05, 0.1) for k in NEEDS}})
    assigned = tiers.assign(scores)
    assert assigned["tier"].to_list() == ["self_serve"]
    assert assigned["tier_rule"].to_list() == [None]


def test_a_rule_needs_every_cell_of_its_row_to_hold() -> None:
    """Each rule is one row of a table, and the cells are ANDed. Half a row is not a rule."""
    cohort = _cohort([{"veteran_id": "v", "med_controlled": True, "n_active_meds": 3}])
    scores = _scores({"v": {k: (0.05, 0.1) for k in NEEDS}})
    # The veteran half holds, the station half does not.
    assert _tier(_assign(scores, cohort, _hazards([{}]), _sites([{}])), "v")[1] is None
    # And the other way round: the station is down, but this veteran is not the one at risk.
    plain = _cohort([{"veteran_id": "v"}])
    down = _sites([{"site_down": True}])
    assert _tier(_assign(_scores({"v": {k: (0.05, 0.1) for k in NEEDS}}), plain,
                         _hazards([{}]), down), "v")[1] is None


def test_a_missing_hazard_row_is_not_a_hazard() -> None:
    """A ZIP or station with no row for the day must read as calm, never as null-is-true."""
    cohort = _cohort([{"veteran_id": "v-dialysis", "ckd_dialysis": True,
                       "modzcta": "11215", "facility_id": "526"}])
    scores = _scores({"v-dialysis": {k: (0.05, 0.1) for k in NEEDS}})
    assigned = _assign(scores, cohort, _hazards([{}]), _sites([{"site_down": True}]))
    assert _tier(assigned, "v-dialysis") == ("self_serve", None), (
        "station 526 has no row on this day, so nothing is known to be closed")


def test_the_rules_run_over_the_real_panel_and_fire_on_somebody() -> None:
    """A rule that cannot fire on the suite's own panel is not being tested by anything."""
    scores, cohort = table("scores"), table("cohort")
    assigned = tiers.assign(scores, cohort=cohort, hazards=table("hazards"),
                            site_status=table("site_status"))
    assert assigned.height == scores.select("veteran_id", "date").unique().height
    fired = (assigned.filter(pl.col("tier_rule").is_not_null())
                     .group_by("tier_rule").len().sort("tier_rule"))
    assert fired.height, "not one of the four rules fired anywhere on the panel"
    assert (assigned.filter(pl.col("tier_rule").is_not_null())["tier"] == "act_now").all(), (
        "a veteran a rule fired on is Act-now by definition")


# --------------------------------------------------------------------------- #
# Part 2: the tier priority band
# --------------------------------------------------------------------------- #

def test_the_band_orders_act_now_before_find_out_before_self_serve() -> None:
    assert tiers.BAND["act_now"] < tiers.BAND["find_out"] < tiers.BAND["self_serve"] \
        < tiers.BAND["everyday"], "SPEC §7.4: the order the deck promises"
    assert set(tiers.BAND) == set(TIERS), "every tier needs a band, or a veteran has none"


def test_one_scarce_call_goes_to_the_act_now_veteran_not_the_richer_self_serve_one() -> None:
    """The prompt-10 symptom, in five rows.

    `v-self` carries more total harm and would win a pure greedy-by-EHA race. `v-act` is
    the one the badge says to call today. With one call in the building, the band decides.
    """
    cohort = _cohort([{"veteran_id": "v-act"}, {"veteran_id": "v-self"}])
    scores = _scores({
        # Act-now: sharp risk on one heavy need, narrow interval.
        "v-act": {"treatment_gap": (0.30, 0.1)},
        # Self-serve: broad moderate risk, worth more EHA in total, below every Act-now line.
        "v-self": {k: (0.24, 0.1) for k in NEEDS},
    })
    cap = dict.fromkeys(DEFAULT_CAPACITY, 0) | {"call": 1}

    graded = tiers.assign(scores)
    assert dict(graded.select("veteran_id", "tier").iter_rows()) == {
        "v-act": "act_now", "v-self": "self_serve"}, "the case has to be the case"

    acted = allocate(scores, cohort, cap, lead=NO_LEAD)
    assert acted.height == 1
    assert acted["veteran_id"][0] == "v-act", (
        "the one call must go to the Act-now veteran even though the Self-serve veteran's "
        f"action averts more on its own; got {acted.select('veteran_id', 'tier', 'eha')}")


def test_no_self_serve_veteran_holds_a_scarce_slot_an_act_now_veteran_wanted() -> None:
    """(c), over the whole panel: the invariant the band exists to enforce.

    Read per veteran per work day, because that is the unit the band is applied at and the
    unit a care team experiences: a veteran is Act-now *today* if any of the risk days being
    worked for them today makes them Act-now. The claim is the prompt's own -- nobody at a
    worse band holds a scarce slot while somebody at a better one, who could have used that
    same bucket, was sent away with nothing at all.
    """
    hazards, sites = table("hazards"), table("site_status")
    scores, cohort = table("scores"), table("cohort")
    kw = dict(hazards=hazards, site_status=sites)
    actions = allocate(scores, cohort, DEFAULT_CAPACITY, **kw)
    # Who *could* have used each bucket: whoever receives it when nothing is scarce. A
    # veteran still gets only their own slots' worth, so this is a subset of the true
    # candidate list -- which can only make the check below more forgiving, never wrong.
    plenty = allocate(scores, cohort, dict.fromkeys(DEFAULT_CAPACITY, 10_000), **kw)

    band = _band_by_work_day(
        tiers.assign(scores, cohort=cohort, hazards=hazards, site_status=sites), actions)
    served = set(actions.select("date", "veteran_id").iter_rows())
    could = {key: set(g["veteran_id"])
             for key, g in plenty.group_by("date", "capacity_bucket")}

    offenders = []
    for (day, bucket), grp in actions.group_by("date", "capacity_bucket"):
        if bucket == "free" or grp.height < DEFAULT_CAPACITY[bucket]:
            continue                      # the bucket never filled; nobody was turned away
        holding = max(band[day, vid] for vid in grp["veteran_id"])
        turned_away = [vid for vid in could.get((day, bucket), ())
                       if (day, vid) not in served]
        if turned_away and min(band[day, vid] for vid in turned_away) < holding:
            offenders.append((str(day), bucket, holding, len(turned_away)))
    assert not offenders, (
        "a full scarce bucket was held by a veteran at a worse band than someone who could "
        f"have used that same bucket and went home with nothing at all: {offenders[:5]}")


def _band_by_work_day(graded: pl.DataFrame, actions: pl.DataFrame) -> dict:
    """(work day, veteran) -> the band they were worked at: the best of the risk days that
    day reaches, which is the same rule `allocate()` applies."""
    by_risk_day = {(vid, day): tiers.BAND[tier]
                   for vid, day, tier in graded.select("veteran_id", "date", "tier").iter_rows()}
    worst = max(tiers.BAND.values())
    out: dict[tuple, int] = {}
    for day, vid, lead in actions.select("date", "veteran_id", "lead_days").iter_rows():
        b = by_risk_day.get((vid, day + timedelta(days=lead)), worst)
        out[day, vid] = min(out.get((day, vid), b), b)
    for (vid, risk_day), b in by_risk_day.items():          # days they were offered nothing
        for lead in range(8):
            key = (risk_day - timedelta(days=lead), vid)
            out[key] = min(out.get(key, b), b)
    return out


def test_the_band_costs_less_than_fifteen_percent_of_the_harm_it_used_to_avert() -> None:
    """(d). The band is a deliberate trade: the badge becomes true, and it is not free."""
    scores, cohort = table("scores"), table("cohort")
    actions = allocate(scores, cohort, DEFAULT_CAPACITY)
    now_total = total_eha(actions)
    now_scarce = total_eha(actions.filter(pl.col("capacity_bucket") != "free"))

    for label, now, before in (("total", now_total, BEFORE_TOTAL_EHA),
                               ("scarce-bucket", now_scarce, BEFORE_SCARCE_EHA)):
        lost = (before - now) / before
        assert lost <= MAX_EHA_LOSS, (
            f"the tier band cost {lost:.1%} of {label} harm averted ({before:,.1f} -> "
            f"{now:,.1f}); the bar is {MAX_EHA_LOSS:.0%}. Either the band is reaching "
            f"further than act_now-before-the-rest, or the rules are firing too widely.")


@pytest.mark.parametrize("bucket", ["call", "ride", "booking", "evac", "pharmacist_slot",
                                    "va_fill", "refill"])
def test_the_band_does_not_break_capacity_monotonicity(bucket: str) -> None:
    """The band is applied per veteran per work day, and that is *why* this still holds.

    Band a candidate and the greedy stops running in value order, so a cheap Act-now action
    can take the slot a far more valuable Self-serve one would have used and the total can
    fall when capacity rises. Band the *veteran* and it cannot: a veteran's own candidates
    are still tried best-first, so every action displaced by a newly affordable one belongs
    to the same veteran and is worth no more than it.
    """
    scores, cohort = table("scores"), table("cohort")
    hazards, sites = table("hazards"), table("site_status")
    prev = None
    for n in (0, 1, 2, 5, 10, 40):
        got = total_eha(allocate(scores, cohort, dict(DEFAULT_CAPACITY, **{bucket: n}),
                                 hazards=hazards, site_status=sites))
        if prev is not None:
            assert got >= prev - 1e-9, (
                f"{bucket} at {n} averted {got:,.4f}, less than the step below ({prev:,.4f})")
        prev = got


def test_the_baselines_run_under_the_same_band_so_the_gap_is_still_the_ranking() -> None:
    """`compare()` promises only *who goes first* changes. The band is policy, not ranking,
    so it applies to every arm -- otherwise the gap would be measuring two things."""
    scores, cohort = table("scores"), table("cohort")
    _, totals, _ = compare(scores, cohort.with_columns(_age=pl.col("age").cast(pl.Float64)),
                           DEFAULT_CAPACITY, rank_by={"oldest_first": "_age"},
                           hazards=table("hazards"), site_status=table("site_status"))
    assert totals["oldest_first"] > 0
    assert totals["oldest_first"] <= total_eha(
        allocate(scores, cohort, DEFAULT_CAPACITY, hazards=table("hazards"),
                 site_status=table("site_status"))) + 1e-9, (
        "risk-ranking inside the band must still beat working the band oldest-first")


def test_the_card_says_which_rule_fired_not_just_the_tier() -> None:
    """A care team told 'Act now' by a rule needs the mechanism, not a probability sentence."""
    from leeward.api import main

    cohort = _cohort([{"veteran_id": "v", "ckd_dialysis": True}])
    scores = _scores({"v": {k: (0.05, 0.1) for k in NEEDS}})
    assigned = _assign(scores, cohort, _hazards([{}]), _sites([{"site_down": True}]))
    tier, rule = _tier(assigned, "v")

    needs = [main.api.NeedScore(need=k, p_mean=0.05, p_lo80=0.035, p_hi80=0.065,
                               p_epistemic_share=0.1, drivers=[], driver_contribs=[])
             for k in NEEDS]
    why = main._why_tier(tier, needs, severity.load(), rule)
    assert "5%" not in why, (
        "a rule-triggered Act-now card must not explain itself with the probability that "
        f"did not trigger it; got {why!r}")
    assert why.strip().endswith("."), "one readable sentence"
    assert len(why) > 40, f"the sentence has to carry the mechanism; got {why!r}"

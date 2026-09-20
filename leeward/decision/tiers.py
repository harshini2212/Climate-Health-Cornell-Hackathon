"""Tier per veteran-day (SPEC §7.5), from the scores and the four hazard rules.

    act_now     p_mean >= 0.25 on a need with w >= 4, and that need's epistemic share < 0.4,
                **or** any row of `act_now.yaml` matches
    find_out    epistemic share >= 0.4 and p_mean >= 0.10 on any need
    self_serve  otherwise, if p_mean >= 0.05 on any need
    everyday    the rest

The SPEC gives self-serve as p_mean 0.05-0.25, which leaves a veteran at 0.25 or more on
only a w = 3 need (breathing, access loss) in no tier. Here they are self-serve.

**A hazard rule outranks the probability.** The four rows of `act_now.yaml` -- site-dependent
care at a closed station, a controlled substance at a closed station, a short mail-order
supply on a delivery-disrupted day, powered equipment with no caregiver in an outage -- each
encode a mechanism the posterior has never seen. A veteran who matches one is Act-now
whatever their p_mean says, and `tier_rule` names which row it was, so the care team is told
the mechanism rather than a probability that did not trigger anything. See `rules.py`; the
YAML is the specification, not a copy of it.

The rules need `cohort`, `hazards` and `site_status`. A caller who asks for neither hazards
nor site status gets the score-only tiers, which is the honest answer to "what do the
probabilities alone say" and is what `eval/fairness.py` asks for. A caller who offers one of
them and not the rest is refused, because a half-applied rule table would quietly stop some
of the four firing and still look like a full answer.

`BAND` is the priority order those tiers give the allocator (SPEC §7.4). It lives here
because it is a statement about tiers; what it does with them is `allocate.py`'s business.
"""

from __future__ import annotations

import polars as pl

from leeward.decision import rules as rule_table
from leeward.decision import severity
from leeward.schema import TIERS

ACT_NOW_P = 0.25
ACT_NOW_MIN_WEIGHT = 4
EPISTEMIC_CUT = 0.4
FIND_OUT_P = 0.10
SELF_SERVE_P = 0.05

#: Which tier gets a scarce slot first (SPEC §7.4). Lower goes first; EHA ranks within a
#: band. Before this existed, allocation was pure greedy-by-EHA and a Self-serve veteran
#: with broad moderate risk outranked an Act-now veteran with one sharp risk for the same
#: call, so the tier badge promised a call the list never made.
BAND = {tier: i for i, tier in enumerate(TIERS)}


def assign(scores: pl.DataFrame, weights: dict[str, float] | None = None, *,
           cohort: pl.DataFrame | None = None, hazards: pl.DataFrame | None = None,
           site_status: pl.DataFrame | None = None,
           rules: dict[str, rule_table.Rule] | None = None) -> pl.DataFrame:
    """veteran_id, date, tier, tier_rule -- one row per veteran-day in `scores`.

    `tier_rule` is the `act_now.yaml` row that forced the tier, or null where the
    probabilities decided it on their own.
    """
    w = weights if weights is not None else severity.load()
    heavy = [k for k, v in w.items() if v >= ACT_NOW_MIN_WEIGHT]
    p, share = pl.col("p_mean"), pl.col("p_epistemic_share")
    per = scores.group_by("veteran_id", "date").agg(
        (pl.col("need").is_in(heavy) & (p >= ACT_NOW_P) & (share < EPISTEMIC_CUT))
        .any().alias("act"),
        ((share >= EPISTEMIC_CUT) & (p >= FIND_OUT_P)).any().alias("find"),
        p.max().alias("p_max"),
    )
    per = _rules(per, cohort, hazards, site_status, rules)
    return per.select(
        "veteran_id", "date",
        pl.when(pl.col("tier_rule").is_not_null() | pl.col("act")).then(pl.lit("act_now"))
          .when("find").then(pl.lit("find_out"))
          .when(pl.col("p_max") >= SELF_SERVE_P).then(pl.lit("self_serve"))
          .otherwise(pl.lit("everyday")).alias("tier"),
        "tier_rule",
    ).sort("date", "veteran_id")


def _rules(vet_days: pl.DataFrame, cohort: pl.DataFrame | None,
           hazards: pl.DataFrame | None, site_status: pl.DataFrame | None,
           rules: dict[str, rule_table.Rule] | None) -> pl.DataFrame:
    """`vet_days` with `tier_rule`: which hazard rule fired, or null throughout when the
    caller asked for the score-only tiers."""
    if hazards is None and site_status is None:
        return vet_days.with_columns(tier_rule=pl.lit(None, dtype=pl.Utf8))
    missing = [name for name, frame in (("cohort", cohort), ("hazards", hazards),
                                        ("site_status", site_status)) if frame is None]
    if missing:
        raise ValueError(
            f"tiers.assign() was asked for the hazard rules but not given {missing}. The "
            f"four Act-now rules in {rule_table.PATH.name} read all three tables, so a "
            f"partial set would silently stop some of them firing. Pass cohort, hazards and "
            f"site_status together, or leave the hazard frames out for score-only tiers.")
    return rule_table.fired(vet_days, cohort, hazards, site_status, rules)

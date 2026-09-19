"""Tier per veteran-day (SPEC §7.5), from the scores alone.

    act_now     p_mean >= 0.25 on a need with w >= 4, and that need's epistemic share < 0.4
    find_out    epistemic share >= 0.4 and p_mean >= 0.10 on any need
    self_serve  otherwise, if p_mean >= 0.05 on any need
    everyday    the rest

The SPEC gives self-serve as p_mean 0.05-0.25, which leaves a veteran at 0.25 or more on
only a w = 3 need (breathing, access loss) in no tier. Here they are self-serve.

Not yet implemented: the hazard-triggered act-now rules (site-dependent × SiteDown; no
caregiver + powered equipment + outage; mail-order supply short on a delivery-disrupted
day; controlled substance × SiteDown). They need hazards and site_status, which
`allocate()` does not take yet.
"""

from __future__ import annotations

import polars as pl

from leeward.decision import severity

ACT_NOW_P = 0.25
ACT_NOW_MIN_WEIGHT = 4
EPISTEMIC_CUT = 0.4
FIND_OUT_P = 0.10
SELF_SERVE_P = 0.05


def assign(scores: pl.DataFrame, weights: dict[str, float] | None = None) -> pl.DataFrame:
    """veteran_id, date, tier -- one row per veteran-day in `scores`."""
    w = weights if weights is not None else severity.load()
    heavy = [k for k, v in w.items() if v >= ACT_NOW_MIN_WEIGHT]
    p, share = pl.col("p_mean"), pl.col("p_epistemic_share")
    per = scores.group_by("veteran_id", "date").agg(
        (pl.col("need").is_in(heavy) & (p >= ACT_NOW_P) & (share < EPISTEMIC_CUT))
        .any().alias("act"),
        ((share >= EPISTEMIC_CUT) & (p >= FIND_OUT_P)).any().alias("find"),
        p.max().alias("p_max"),
    )
    return per.select(
        "veteran_id", "date",
        pl.when("act").then(pl.lit("act_now"))
          .when("find").then(pl.lit("find_out"))
          .when(pl.col("p_max") >= SELF_SERVE_P).then(pl.lit("self_serve"))
          .otherwise(pl.lit("everyday")).alias("tier"),
    ).sort("date", "veteran_id")

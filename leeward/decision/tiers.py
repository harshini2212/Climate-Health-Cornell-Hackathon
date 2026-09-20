"""Tier per veteran-day (SPEC §7.5), from the scores alone.

    act_now     p_mean >= 0.25 on a need with w >= 4, and that need's epistemic share < 0.4
    find_out    epistemic share >= 0.4 and p_mean >= 0.10 on any need, OR a gap in this
                veteran's record that straddles a line the care team acts on
    self_serve  otherwise, if p_mean >= 0.05 on any need
    everyday    the rest

The second find-out rule is the one that earns the three-minute call, and it is about this
veteran's record rather than the model. `model/score_prior.py` reports `p_gap_lo` and
`p_gap_hi`: the p_mean this veteran would be reported with if the fields the VA does not
have on file turned out to be their least- and most-risky values. When that range straddles
a threshold below -- when the answer decides whether the veteran is someone the team spends
a call or an intervention on -- the call is worth three minutes, because it changes what
happens next. When the range sits entirely on one side, the answer is interesting and not
actionable, and the team's afternoon is better spent elsewhere.

Why not the epistemic share alone. `p_epistemic_share` is Var(p) relative to p(1-p), and at
rung 0 it is dominated by the prior spread every veteran shares, not by anything specific to
one of them: across 6,000,000 scored rows it reaches 0.426 against a 0.4 cut, so the first
rule fires on 0.04 percent of veteran-days. Lowering the cut does not help either, because
act-now requires the share to be *below* the same number -- at a 0.25 cut find-out reaches
3.1 percent and act-now collapses to 0.004. The two rules were competing for one threshold.
The gap rule does not compete: it reads a different column.

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


#: The lines a gap has to straddle to be worth asking about: the two that gate a scarce
#: resource. Crossing the self-serve line only changes a text message, which is not worth
#: a person's afternoon, so `SELF_SERVE_P` is deliberately not in here.
ASKABLE_LINES = (FIND_OUT_P, ACT_NOW_P)


def _worth_asking(heavy: list[str], has_gap_columns: bool) -> pl.Expr:
    """Would filling this record's gap change which side of a line the veteran is on?

    False everywhere when the scores predate `p_gap_lo`/`p_gap_hi` -- an older parquet or a
    lane's own fixture -- so this never invents a tier out of a missing column.
    """
    if not has_gap_columns:
        return pl.lit(False)
    lo, hi = pl.col("p_gap_lo"), pl.col("p_gap_hi")
    straddles = pl.any_horizontal([(lo < line) & (hi >= line) for line in ASKABLE_LINES])
    return pl.col("need").is_in(heavy) & straddles.fill_null(False)


def assign(scores: pl.DataFrame, weights: dict[str, float] | None = None) -> pl.DataFrame:
    """veteran_id, date, tier -- one row per veteran-day in `scores`."""
    w = weights if weights is not None else severity.load()
    heavy = [k for k, v in w.items() if v >= ACT_NOW_MIN_WEIGHT]
    p, share = pl.col("p_mean"), pl.col("p_epistemic_share")
    gap_cols = {"p_gap_lo", "p_gap_hi"} <= set(scores.columns)
    per = scores.group_by("veteran_id", "date").agg(
        (pl.col("need").is_in(heavy) & (p >= ACT_NOW_P) & (share < EPISTEMIC_CUT))
        .any().alias("act"),
        (((share >= EPISTEMIC_CUT) & (p >= FIND_OUT_P)) | _worth_asking(heavy, gap_cols))
        .any().alias("find"),
        p.max().alias("p_max"),
    )
    return per.select(
        "veteran_id", "date",
        pl.when("act").then(pl.lit("act_now"))
          .when("find").then(pl.lit("find_out"))
          .when(pl.col("p_max") >= SELF_SERVE_P).then(pl.lit("self_serve"))
          .otherwise(pl.lit("everyday")).alias("tier"),
    ).sort("date", "veteran_id")

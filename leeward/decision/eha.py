"""Expected harm averted (SPEC §7.3).

    EHA[i,a,t] = Σ_k  w_k · p_mean[i,k,t] · τ[a,k]
    VOI[i,t]   = Σ_k  w_k · epistemic_var[i,k,t]          the value of a check-in call

`epistemic_var` comes back exactly from the scores table. Over posterior draws,
Var(p) + E[p(1-p)] = p̄(1-p̄), so the stored share s = Var / (Var + E[p(1-p)]) gives
Var = s · p̄(1-p̄).

VOI is the variance form from the SPEC §7.2 table ("expected reduction in epistemic variance
× w"), not the square root written in §7.3. On the fixtures the square-root form is about five
times a care-team call's EHA and would hand every call slot to a check-in; the variance form
sits on the same scale as the harm it competes with.

An action only has an EHA where it applies. `ELIGIBLE` says who can receive what: booking an
alternate dialysis site for someone who is not on dialysis is not a low-value action, it is
a wrong one.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from leeward.decision import severity
from leeward.decision import tau as tau_table
from leeward.schema import NEEDS

_ANYONE = pl.col("veteran_id").is_not_null()
_ON_MEDS = pl.col("n_active_meds") > 0
_SITE_DEPENDENT = pl.col("ckd_dialysis") | pl.col("active_cancer_tx") | pl.col("on_methadone_otp")
_MED_SIGNAL = ((pl.col("med_thermoreg_score") > 0) | (pl.col("acb_score") >= 3)
               | pl.col("med_combo_raas_diuretic") | pl.col("med_renal_triple")
               | pl.col("med_narrow_ti"))

#: action -> which cohort rows it can go to. Tier gating (Everyday gets nothing, check-ins are
#: for Find-out) is applied by allocate.py, because tiers change daily and the cohort does not.
ELIGIBLE: dict[str, pl.Expr] = {
    "care_team_call": _ANYONE,
    "check_in_call": _ANYONE,
    "verified_text": _ANYONE,
    "cooling_center_ride": _ANYONE,
    "clean_air_room": _ANYONE,
    "early_refill": _ON_MEDS,
    "switch_to_local_pickup": _ON_MEDS & pl.col("mail_order_pharmacy"),
    "pharmacist_med_review": _MED_SIGNAL,
    "cold_chain_plan": pl.col("med_cold_chain"),
    "controlled_substance_bridge": pl.col("med_controlled") | pl.col("on_methadone_otp"),
    "backup_power_plan": pl.col("powered_equipment") != "none",
    "alt_site_booking": _SITE_DEPENDENT,
    "evacuation_assist": pl.col("evac_zone") > 0,
    # SPEC §7.4b availability. Neither has a tau row yet, so neither is ever proposed.
    "assign_buddy": pl.col("caregiver") == "none",
    "heap_application": pl.col("low_assets"),
}

#: Every cohort column the rules above read.
COHORT_COLUMNS = sorted({c for e in ELIGIBLE.values() for c in e.meta.root_names()}
                        - {"veteran_id"})

#: Per-need fields carried from scores into the wide frame.
FIELDS = ("p_mean", "p_lo80", "p_hi80", "p_epistemic_share", "driver_1")


def needs_wide(scores: pl.DataFrame) -> pl.DataFrame:
    """One row per veteran-day, with a `<field>_<need>` column for every field and need."""
    wide = scores.pivot(on="need", index=["veteran_id", "date"], values=list(FIELDS))
    p_cols = [f"p_mean_{k}" for k in NEEDS]
    absent = [c for c in p_cols if c not in wide.columns]
    if absent or wide.select(pl.any_horizontal(pl.col(p_cols).is_null()).any()).item():
        raise ValueError("scores is missing needs for some veteran-days; every veteran-day "
                         f"needs all of {NEEDS} (absent columns: {absent})")
    return wide


def epistemic_var(p, share):
    """Variance of p over posterior draws, from the mean and the stored epistemic share."""
    return share * p * (1 - p)


def eha_matrix(p: np.ndarray, weights: dict[str, float],
               table: dict[str, dict[str, float]]) -> tuple[list[str], np.ndarray]:
    """Standalone EHA for every prevention action: (actions, n x A). `p` is n x K, NEEDS order."""
    actions, T = tau_table.matrix(table)
    return actions, (p * severity.vector(weights)) @ T.T


def voi(p: np.ndarray, share: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """Value of a check-in call per row: weighted epistemic variance summed over needs."""
    return (severity.vector(weights) * epistemic_var(p, share)).sum(axis=1)

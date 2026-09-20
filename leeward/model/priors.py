"""Priors on every coefficient in `design.py`. Rung 0 scores straight from these.

Each entry names a term in `design.TERMS`. A dict gives one Normal per need; a bare
`Prior` is **one** parameter shared by every need the term acts on. Variable names in
`data/posterior.nc` match the keys here, at every rung.

`source` says where each number comes from: "SPEC 6.3" where docs/SPEC.md fixed it, or a
one-line reason where it did not. None of these are measurements. They are the modelling
assumptions a fit is supposed to move, and at rung 0 they are all there is -- so they are
written down where a judge can ask about them.

One deliberate departure from SPEC 6.3: the spec centres the heat and PM2.5 lag curves on
zero. At rung 0 the prior mean *is* the forecast, so a zero-centred curve would say a heat
wave does not raise heat risk. Those two curves are centred on a positive, decaying shape
instead. Rung 1 may re-centre them once there is data to pull them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from leeward.model import design
from leeward.schema import NEEDS


@dataclass(frozen=True)
class Prior:
    """Normal(mean, sd) on a coefficient, in log-odds per unit of its feature."""
    mean: float | tuple[float, ...]    # one per lag for a lag curve
    sd: float
    source: str


def _each(needs: list[str], mean, sd: float, source: str) -> dict[str, Prior]:
    return {n: Prior(mean, sd, source) for n in needs}


_ASSUME = "assumption: weakly informative, positive"

PRIORS: dict[str, Prior | dict[str, Prior]] = {
    "alpha": _each(NEEDS, -5.5, 1.0, "SPEC 6.2: Normal(-5.5, 1); ~0.4%/day, the simulator's "
                                      "0.2-2% off-event band"),

    # health
    "beta_age65": {"heat": Prior(0.4, 0.3, _ASSUME),
                   "treatment_gap": Prior(0.2, 0.3, _ASSUME)},
    "beta_copd": {"breathing": Prior(0.8, 0.4, _ASSUME + "; the main respiratory condition")},
    "beta_asthma": {"breathing": Prior(0.6, 0.4, _ASSUME)},
    "beta_chf": {"breathing": Prior(0.4, 0.4, _ASSUME), "heat": Prior(0.3, 0.3, _ASSUME)},
    "beta_diabetes": {"heat": Prior(0.2, 0.3, _ASSUME),
                      "treatment_gap": Prior(0.2, 0.3, _ASSUME)},
    "beta_dialysis": {"treatment_gap": Prior(0.8, 0.5, _ASSUME + "; thrice-weekly care on site")},
    "beta_cancer_tx": {"treatment_gap": Prior(0.5, 0.4, _ASSUME + "; scheduled infusion")},
    "beta_ptsd": {"mental": Prior(0.3, 0.3, "SPEC 6.3: WTC mind-body")},
    "beta_depression": {"mental": Prior(0.5, 0.4, _ASSUME)},
    "beta_mobility": {"access_loss": Prior(0.4, 0.4, _ASSUME)},

    # social and resources
    "sigma_no_caregiver": _each(["heat", "mental", "treatment_gap", "access_loss"], 0.5, 0.4,
                                "SPEC 6.3: clinician input; social-isolation literature"),
    "sigma_low_assets": _each(["heat", "treatment_gap", "access_loss"], 0.4, 0.4,
                              "SPEC 6.3: NYC heat report, AC owned but not run for cost"),

    # hazards, main effects
    "delta_heat": {"heat": Prior((0.5, 0.3, 0.2, 0.1), 0.3,
                                 "SPEC 6.3 centres lags at 0; recentred for rung 0, see module "
                                 "docstring. Per 10F above NYC Health's 82F threshold")},
    "eps_pm25": {"breathing": Prior((0.4, 0.2, 0.1), 0.3,
                                    "SPEC 6.3 centres lags at 0; recentred for rung 0. Per "
                                    "50 ug/m3 above EPA's 24-hour standard")},
    "zeta_flood": {"treatment_gap": Prior(0.3, 0.4, _ASSUME),
                   "access_loss": Prior(0.6, 0.5, _ASSUME)},
    "zeta_evac": {"treatment_gap": Prior(0.4, 0.4, _ASSUME),
                  "access_loss": Prior(0.8, 0.5, _ASSUME + "; an order moves people")},
    "kappa_outage": {"breathing": Prior(0.2, 0.3, _ASSUME),
                     "treatment_gap": Prior(0.4, 0.4, _ASSUME)},
    "psi_sitedown": {"treatment_gap": Prior(0.5, 0.5, _ASSUME + "; every patient of a closed site"),
                     "access_loss": Prior(0.4, 0.4, _ASSUME)},

    # interactions -- medication x heat, per CDC's clinician guidance
    "theta_heat_x_meds": {"heat": Prior(0.35, 0.3, "SPEC 6.3: CDC mechanism list; per score unit")},
    "theta_heat_x_raas_diuretic": {"heat": Prior(0.5, 0.4, "SPEC 6.3: CDC names this pair")},
    "theta_heat_x_acb": {"heat": Prior(0.5, 0.4, "SPEC 6.3: ACB >= 3 threshold")},
    "theta_heat_x_renal_triple": {"heat": Prior(0.6, 0.5, "SPEC 6.3: AKI with dehydration")},
    "theta_heat_x_no_ac": {"heat": Prior(0.6, 0.4, "assumption; NYC Health: no home cooling is "
                                                   "the dominant heat-death risk factor")},
    "theta_low_assets_x_heat": {"heat": Prior(0.5, 0.4, "SPEC 6.3")},

    # interactions -- the rest
    "theta_smoke_x_pact": {"breathing": Prior(0.6, 0.5, "SPEC 6.3: PACT presumptive list")},
    "theta_outage_x_equipment": {"treatment_gap": Prior(0.8, 0.6, "SPEC 6.3: emPOWER rationale")},
    "theta_outage_x_cold_chain": {"treatment_gap": Prior(0.9, 0.6,
                                                         "SPEC 6.3: insulin spoils in about a day")},
    "theta_flood_x_lowfloor": {"access_loss": Prior(0.8, 0.6, "SPEC 6.3: Ida basement deaths")},
    "theta_flood_x_mobility": {"access_loss": Prior(0.5, 0.5, _ASSUME)},
    "theta_sitedown_x_sitedependent": {"treatment_gap": Prior(1.0, 0.7,
                                                              "SPEC 6.3: Sandy dialysis/OTP")},
    "theta_sitedown_x_controlled": {"treatment_gap": Prior(0.9, 0.6,
                                                           "SPEC 6.3: retail refill excludes "
                                                           "controlled substances")},
    "theta_mail_x_supply": {"treatment_gap": Prior(1.2, 0.6, "SPEC 6.3: near-deterministic; "
                                                             "wide anyway so it can shrink")},
    "theta_no_caregiver_x_hazard": Prior(0.4, 0.4, "SPEC 6.3: one parameter across hazards"),
    "theta_low_assets_x_flood_outage": Prior(0.5, 0.4, "SPEC 6.3"),
}


def per_need(name: str) -> dict[str, Prior]:
    """The term's prior, one entry per need it acts on -- a shared prior repeated."""
    spec = PRIORS[name]
    return dict.fromkeys(design.TERM_BY_NAME[name].needs, spec) if isinstance(spec, Prior) else spec


def _validate() -> None:
    """Every design cell has exactly one prior, and every prior lands on a design cell.

    A cell with no prior would be a term the simulator can switch on and the model cannot
    see, which is a silent misspecification. So it is an import-time error instead.
    """
    unknown = set(PRIORS) - set(design.TERM_BY_NAME)
    if unknown:
        raise ValueError(f"priors for terms design.py does not have: {sorted(unknown)}")
    for t in design.TERMS:
        if t.name not in PRIORS:
            raise ValueError(f"design term {t.name} has no prior")
        spec = per_need(t.name)
        if set(spec) != set(t.needs):
            raise ValueError(f"{t.name}: priors cover {sorted(spec)}, the term acts on "
                             f"{sorted(t.needs)}")
        for need, p in spec.items():
            if np.atleast_1d(p.mean).shape != (len(t.exprs),):
                raise ValueError(f"{t.name}[{need}]: {len(t.exprs)} lag means expected")
            if not p.sd > 0:
                raise ValueError(f"{t.name}[{need}]: sd must be positive")


_validate()


def mean_matrix() -> np.ndarray:
    """(P, K) prior means; zero off the design's support."""
    return design.coef_matrix({name: {n: p.mean for n, p in per_need(name).items()}
                               for name in PRIORS})


def support_mask() -> np.ndarray:
    """(P, K) True where a coefficient has a prior."""
    return design.SUPPORT.copy()


def draw(n_draws: int, *, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """(D, P, K) coefficient draws. Cells without a prior are exactly zero.

    `scale` is SPEC 6.3's prior_scale_multiplier (0.5 / 1 / 2 on the demo slider): it
    multiplies every sd and leaves every mean alone. Draw order is fixed, so a seed
    reproduces the same coefficients on any machine.
    """
    rng = np.random.default_rng(seed)
    B = np.zeros((n_draws, design.P, design.K))
    for name, spec in PRIORS.items():
        rows = design.TERM_SLICE[name]
        n_feat = rows.stop - rows.start
        if isinstance(spec, Prior):
            shared = np.asarray(spec.mean) + scale * spec.sd * rng.standard_normal((n_draws, n_feat))
            for need in design.TERM_BY_NAME[name].needs:
                B[:, rows, NEEDS.index(need)] = shared
            continue
        for need in NEEDS:
            if need in spec:
                p = spec[need]
                z = rng.standard_normal((n_draws, n_feat))
                B[:, rows, NEEDS.index(need)] = np.asarray(p.mean) + scale * p.sd * z
    return B

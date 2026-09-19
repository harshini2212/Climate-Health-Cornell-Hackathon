"""The one feature mapping, shared by the simulator and every model rung.

Frames in, arrays out, no I/O. `cohort/simulate.py` builds `X` here and multiplies it by
`truth.json`; the model builds the same `X` and multiplies it by prior or posterior draws.
One mapping, so the world we simulate and the world we fit cannot drift apart.

    d   = design.build(cohort, hazards, site_status)   # one row per veteran x date
    B   = design.coef_matrix(truth)                     # (P features, K needs)
    eta = design.linear_predictor(d.X, B)               # (N, K) log-odds
    eta = design.linear_predictor(d.X, draws)           # draws (D, P, K) -> (N, K, D)

The linear predictor is exactly `X @ B`. The intercept is a column of ones, so `alpha` is
one more row of B. Anything the model cannot see -- the latent burn-pit dose, a ZIP effect,
per-veteran frailty -- belongs to the simulator and enters through `offset`.

A coefficient is named after its term and, for the lag curves, indexed by lag:

    {"alpha":        {"heat": -6.0, ...},
     "delta_heat":   {"heat": [0.9, 0.7, 0.4, 0.15]},      # lags 0..3
     "theta_no_caregiver_x_hazard": 0.5}                    # scalar = every need it acts on

Units, chosen so a coefficient reads as log-odds per something a clinician recognises:

    heat     per 10 F of heat index above 82 F   NYC Health's non-extreme-hot-day threshold
    PM2.5    per 50 ug/m3 above 35               EPA's 24-hour PM2.5 standard
    outage   fraction of the ZIP without power, 0-1
    hot day  heat index >= 82 F, `hazards.hot_day`
    med_thermoreg_score is per score unit; every other feature is 0/1
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from leeward.schema import NEEDS

HEAT_HINGE_F = 82.0     # NYC Health 2026 heat mortality report; docs/sources.md
HEAT_UNIT_F = 10.0
HEAT_CAP = 2.5          # 107 F. Guards against a typo in a scenario, not a real heat wave.
HEAT_LAGS = 4

PM25_STANDARD = 35.0    # EPA 24-hour PM2.5 NAAQS, ug/m3; docs/sources.md
PM25_UNIT = 50.0
PM25_CAP = 4.0          # 235 ug/m3, above the real 7 June 2023 peak of 203.5
PM25_LAGS = 3

FILL_DAYS_MAIL = 90     # VA's 90-day mail fill and 30-day window fill (data/README.md)
FILL_DAYS_RETAIL = 30
SUPPLY_SHORT_DAYS = 7   # assumption: under a week of medication left counts as short

K = len(NEEDS)
_ALL = tuple(NEEDS)


def _f(expr: pl.Expr) -> pl.Expr:
    return expr.cast(pl.Float64)


# --------------------------------------------------------------------------- #
# Building blocks. Columns come from cohort, from _hazard_frame, or from site_status.
# --------------------------------------------------------------------------- #

_NO_CAREGIVER = pl.col("caregiver") == "none"
_SITE_DEPENDENT = pl.col("ckd_dialysis") | pl.col("on_methadone_otp") | pl.col("active_cancer_tx")
_LOW_FLOOR = pl.col("floor").is_in(["basement", "ground"])
_POWERED = pl.col("powered_equipment") != "none"
_EVAC_ORDERED = (pl.col("evac_zone") >= 1) & (pl.col("evac_zone") <= pl.col("evac_zone_ordered"))
_FILL_DAYS = pl.when(pl.col("mail_order_pharmacy")).then(FILL_DAYS_MAIL).otherwise(FILL_DAYS_RETAIL)
# days_supply_remaining is read as of `ref_date`; fills come round on their cycle after that.
_SUPPLY_LEFT = (pl.col("days_supply_remaining") - pl.col("_elapsed")) % _FILL_DAYS
_SUPPLY_SHORT = _SUPPLY_LEFT <= SUPPLY_SHORT_DAYS

_HOT = _f(pl.col("hot_day"))
_SMOKE = _f(pl.col("smoke_day"))
_FLOOD = _f(pl.col("flood"))
_EVAC = _f(_EVAC_ORDERED)
_OUTAGE = pl.col("outage_frac")
_DOWN = _f(pl.col("site_down"))
_MAIL = _f(pl.col("mail_disrupted"))
_ANY_HAZARD = pl.max_horizontal(_HOT, _SMOKE, _FLOOD, _EVAC, _OUTAGE, _DOWN, _MAIL)
_FLOOD_OR_OUTAGE = pl.max_horizontal(_FLOOD, _EVAC, _OUTAGE)


@dataclass(frozen=True, eq=False)
class Term:
    """One block of X with one coefficient per (feature, need) in `needs`."""
    name: str
    #: "intercept" | "person" (static per veteran) | "hazard" (zero on a calm day)
    kind: str
    #: The needs this term may act on. A coefficient anywhere else is a structural zero.
    needs: tuple[str, ...]
    #: One expression per feature; several for a lag curve.
    exprs: tuple[pl.Expr, ...]
    #: Plain language for the driver chip on the veteran card.
    phrase: str
    lagged: bool = False

    @property
    def feature_names(self) -> tuple[str, ...]:
        if not self.lagged:
            return (self.name,)
        return tuple(f"{self.name}[{i}]" for i in range(len(self.exprs)))


def _t(name, kind, needs, expr, phrase) -> Term:
    return Term(name, kind, tuple(needs), (_f(expr),), phrase)


def _lagged(name, needs, column, n, phrase) -> Term:
    return Term(name, "hazard", tuple(needs),
                tuple(pl.col(f"{column}{i}") for i in range(n)), phrase, lagged=True)


TERMS: tuple[Term, ...] = (
    _t("alpha", "intercept", _ALL, pl.lit(1.0), "baseline"),

    # health -- from the record
    _t("beta_age65", "person", ["heat", "treatment_gap"], pl.col("age") >= 65,
       "age 65 or older"),
    _t("beta_copd", "person", ["breathing"], pl.col("copd"), "COPD"),
    _t("beta_asthma", "person", ["breathing"], pl.col("asthma"), "asthma"),
    _t("beta_chf", "person", ["breathing", "heat"], pl.col("chf"), "heart failure"),
    _t("beta_diabetes", "person", ["heat", "treatment_gap"], pl.col("diabetes"), "diabetes"),
    _t("beta_dialysis", "person", ["treatment_gap"], pl.col("ckd_dialysis"), "on dialysis"),
    _t("beta_cancer_tx", "person", ["treatment_gap"], pl.col("active_cancer_tx"),
       "in active cancer treatment"),
    _t("beta_ptsd", "person", ["mental"], pl.col("ptsd"), "PTSD"),
    _t("beta_depression", "person", ["mental"], pl.col("depression"), "depression"),
    _t("beta_mobility", "person", ["access_loss"], pl.col("mobility_impaired"),
       "mobility impairment"),

    # social and resources
    _t("sigma_no_caregiver", "person", ["heat", "mental", "treatment_gap", "access_loss"],
       _NO_CAREGIVER, "no caregiver"),
    _t("sigma_low_assets", "person", ["heat", "treatment_gap", "access_loss"],
       pl.col("low_assets"), "little money for cooling, rides or replacement supplies"),

    # hazards, main effects
    _lagged("delta_heat", ["heat"], "heat_x", HEAT_LAGS,
            "heat index above 82F in the last four days"),
    _lagged("eps_pm25", ["breathing"], "pm_x", PM25_LAGS,
            "PM2.5 above the EPA 24-hour standard in the last three days"),
    _t("zeta_flood", "hazard", ["treatment_gap", "access_loss"], _FLOOD, "flooding in this ZIP"),
    _t("zeta_evac", "hazard", ["treatment_gap", "access_loss"], _EVAC,
       "evacuation ordered for this zone"),
    _t("kappa_outage", "hazard", ["breathing", "treatment_gap"], _OUTAGE,
       "power outage in this ZIP"),
    _t("psi_sitedown", "hazard", ["treatment_gap", "access_loss"], _DOWN,
       "assigned VA site is closed"),

    # interactions -- medication x heat
    _t("theta_heat_x_meds", "hazard", ["heat"], _HOT * pl.col("med_thermoreg_score"),
       "heat-sensitive medications on a hot day"),
    _t("theta_heat_x_raas_diuretic", "hazard", ["heat"],
       _HOT * _f(pl.col("med_combo_raas_diuretic")),
       "ACE inhibitor or ARB plus a diuretic on a hot day"),
    _t("theta_heat_x_acb", "hazard", ["heat"], _HOT * _f(pl.col("acb_score") >= 3),
       "anticholinergic burden of 3 or more on a hot day"),
    _t("theta_heat_x_renal_triple", "hazard", ["heat"], _HOT * _f(pl.col("med_renal_triple")),
       "NSAID with a diuretic and an ACE inhibitor or ARB on a hot day"),
    _t("theta_heat_x_no_ac", "hazard", ["heat"], _HOT * _f(~pl.col("home_ac")),
       "no home air conditioning on a hot day"),
    _t("theta_low_assets_x_heat", "hazard", ["heat"], _HOT * _f(pl.col("low_assets")),
       "may not afford to run cooling on a hot day"),

    # interactions -- the rest of the hazard story
    _t("theta_smoke_x_pact", "hazard", ["breathing"], _SMOKE * _f(pl.col("pact_presumptive")),
       "PACT Act exposure history on a smoke day"),
    _t("theta_outage_x_equipment", "hazard", ["treatment_gap"], _OUTAGE * _f(_POWERED),
       "powered medical equipment during an outage"),
    _t("theta_outage_x_cold_chain", "hazard", ["treatment_gap"],
       _OUTAGE * _f(pl.col("med_cold_chain")), "refrigerated medication during an outage"),
    _t("theta_flood_x_lowfloor", "hazard", ["access_loss"], _FLOOD * _f(_LOW_FLOOR),
       "basement or ground-floor home during flooding"),
    _t("theta_flood_x_mobility", "hazard", ["access_loss"],
       _FLOOD * _f(pl.col("mobility_impaired")), "mobility impairment during flooding"),
    _t("theta_sitedown_x_sitedependent", "hazard", ["treatment_gap"],
       _DOWN * _f(_SITE_DEPENDENT & pl.col("site_dependent_services")),
       "dialysis, infusion or methadone at a closed VA site"),
    _t("theta_sitedown_x_controlled", "hazard", ["treatment_gap"],
       _DOWN * _f(pl.col("med_controlled")),
       "controlled substance, which retail emergency refill excludes, while the VA site is closed"),
    _t("theta_mail_x_supply", "hazard", ["treatment_gap"],
       _MAIL * _f(pl.col("mail_order_pharmacy") & _SUPPLY_SHORT),
       "mail-order medication running out while mail is disrupted"),
    _t("theta_no_caregiver_x_hazard", "hazard", ["heat", "mental", "treatment_gap", "access_loss"],
       _ANY_HAZARD * _f(_NO_CAREGIVER), "no caregiver during a hazard"),
    _t("theta_low_assets_x_flood_outage", "hazard", ["treatment_gap", "access_loss"],
       _FLOOD_OR_OUTAGE * _f(pl.col("low_assets")),
       "little money to ride out a flood or outage"),
)

TERM_BY_NAME: dict[str, Term] = {t.name: t for t in TERMS}
FEATURES: list[str] = [f for t in TERMS for f in t.feature_names]
P = len(FEATURES)


def _slices() -> dict[str, slice]:
    out, i = {}, 0
    for t in TERMS:
        out[t.name] = slice(i, i + len(t.exprs))
        i += len(t.exprs)
    return out


TERM_SLICE: dict[str, slice] = _slices()

#: (P, K) True where a coefficient may be non-zero. Shared by the priors and the simulator.
SUPPORT = np.zeros((P, K), dtype=bool)
for _term in TERMS:
    for _need in _term.needs:
        SUPPORT[TERM_SLICE[_term.name], NEEDS.index(_need)] = True

assert len(TERM_BY_NAME) == len(TERMS), "duplicate term name"
assert all(n in NEEDS for t in TERMS for n in t.needs), "term acts on an unknown need"


# --------------------------------------------------------------------------- #
# The design matrix
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class Design:
    """One row per (veteran, date), cohort order then date order."""
    veteran_id: np.ndarray   # (N,) str
    vet_index: np.ndarray    # (N,) row of the veteran in the cohort frame that was passed in
    date: np.ndarray         # (N,) datetime64[D]
    modzcta: np.ndarray      # (N,) str, for a simulator-side ZIP effect
    X: np.ndarray            # (N, P) float64, columns in FEATURES order


def _hazard_frame(hazards: pl.DataFrame) -> pl.DataFrame:
    """Per ZIP-day hazard features plus lags. A lag before the first date reads 0."""
    h = hazards.select(
        "modzcta", "date",
        heat_x=((pl.col("heat_index_max_f") - HEAT_HINGE_F) / HEAT_UNIT_F).clip(0, HEAT_CAP),
        pm_x=((pl.col("pm25") - PM25_STANDARD) / PM25_UNIT).clip(0, PM25_CAP),
        hot_day=pl.col("hot_day"),
        smoke_day=pl.col("smoke_alert") | (pl.col("pm25") > PM25_STANDARD),
        flood=(pl.col("flood_warning") | pl.col("flash_flood_emergency")
               | pl.col("floodnet_trip") | (pl.col("surge_ft") > 0)),
        evac_zone_ordered=pl.col("evac_zone_ordered"),
        outage_frac=pl.col("outage_frac").cast(pl.Float64),
        mail_disrupted=pl.col("mail_delivery_disrupted"),
    )
    # Lag by calendar date, not by row, so a gap in the table cannot shift a curve.
    for col, n in (("heat_x", HEAT_LAGS), ("pm_x", PM25_LAGS)):
        for lag in range(n):
            shifted = h.select("modzcta", pl.col("date") + pl.duration(days=lag),
                               pl.col(col).alias(f"{col}{lag}"))
            h = h.join(shifted, on=["modzcta", "date"], how="left", maintain_order="left")
    lag_cols = [f"heat_x{i}" for i in range(HEAT_LAGS)] + [f"pm_x{i}" for i in range(PM25_LAGS)]
    return h.drop("heat_x", "pm_x").with_columns(pl.col(lag_cols).fill_null(0.0))


def _feature_exprs() -> list[pl.Expr]:
    return [e.alias(f) for t in TERMS for e, f in zip(t.exprs, t.feature_names, strict=True)]


def _cohort_columns(cohort: pl.DataFrame) -> list[str]:
    used = {c for e in _feature_exprs() for c in e.meta.root_names()}
    keys = ["veteran_id", "modzcta", "facility_id"]
    return keys + sorted((used & set(cohort.columns)) - set(keys))


def _require_joined(rows: pl.DataFrame, probe: str, table: str, keys: list[str]) -> None:
    missing = rows.filter(pl.col(probe).is_null()).select(keys).unique()
    if missing.height:
        sample = ", ".join(str(tuple(r)) for r in missing.head(5).rows())
        raise ValueError(f"{missing.height} {keys} pairs have no {table} row "
                         f"(e.g. {sample}). A missing hazard must not read as no hazard.")


def build(cohort: pl.DataFrame, hazards: pl.DataFrame, site_status: pl.DataFrame, *,
          dates: Iterable[date] | None = None, ref_date: date | None = None) -> Design:
    """The design matrix for every veteran in `cohort` on every date in `dates`.

    `hazards` may cover more days than `dates`; the extra days feed the lag curves.
    `ref_date` is the day `days_supply_remaining` was counted on (default: first hazard day).
    Raises ValueError if any veteran-day has no hazard or site_status row.
    """
    dates = sorted(hazards["date"].unique().to_list()) if dates is None else sorted(dates)
    ref = hazards["date"].min() if ref_date is None else ref_date

    hz = _hazard_frame(hazards)
    sites = site_status.select("facility_id", "date", "site_down", "site_dependent_services")
    rows = (cohort.select(_cohort_columns(cohort)).with_row_index("_i")
            .join(pl.DataFrame({"date": pl.Series(dates, dtype=pl.Date)}).with_row_index("_t"),
                  how="cross")
            .join(hz, on=["modzcta", "date"], how="left")
            .join(sites, on=["facility_id", "date"], how="left")
            .sort("_i", "_t"))

    if rows.height != cohort.height * len(dates):
        raise ValueError("hazards or site_status has duplicate rows on its key")
    _require_joined(rows, "hot_day", "hazards", ["modzcta", "date"])
    _require_joined(rows, "site_down", "site_status", ["facility_id", "date"])

    rows = rows.with_columns(_elapsed=(pl.col("date") - pl.lit(ref, dtype=pl.Date)).dt.total_days())
    X = rows.select(_feature_exprs()).to_numpy().astype(np.float64, copy=False)
    return Design(veteran_id=rows["veteran_id"].to_numpy(), vet_index=rows["_i"].to_numpy(),
                  date=rows["date"].to_numpy(), modzcta=rows["modzcta"].to_numpy(), X=X)


# --------------------------------------------------------------------------- #
# Coefficients, the linear predictor and its decomposition
# --------------------------------------------------------------------------- #

def coef_matrix(params: Mapping[str, float | Mapping[str, float | Sequence[float]]]) -> np.ndarray:
    """Named coefficients -> B of shape (P, K). Unnamed cells are zero.

    Fails loudly on an unknown term, a need the term does not act on, or the wrong number
    of lags, so a typo in truth.json cannot quietly delete a term from the simulator.
    """
    B = np.zeros((P, K))
    for name, value in params.items():
        term = TERM_BY_NAME.get(name)
        if term is None:
            raise ValueError(f"unknown term {name!r}; known terms: {sorted(TERM_BY_NAME)}")
        per_need = value if isinstance(value, Mapping) else dict.fromkeys(term.needs, value)
        for need, v in per_need.items():
            if need not in term.needs:
                raise ValueError(f"{name} acts on {list(term.needs)}, not {need!r}")
            vals = np.atleast_1d(np.asarray(v, dtype=np.float64))
            if vals.shape != (len(term.exprs),):
                raise ValueError(f"{name}[{need}] needs {len(term.exprs)} value(s), got {vals.size}")
            B[TERM_SLICE[name], NEEDS.index(need)] = vals
    return B


def linear_predictor(X: np.ndarray, B: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
    """Log-odds. B (P, K) -> (N, K). Draws B (D, P, K) -> (N, K, D), draws last.

    `offset` is (N,) or (N, K): the simulator's terms the model does not see.
    """
    if B.ndim == 2:
        eta = X @ B
    elif B.ndim == 3:
        D = B.shape[0]
        eta = (X @ B.transpose(1, 2, 0).reshape(P, K * D)).reshape(-1, K, D)
    else:
        raise ValueError(f"B must be (P, K) or (D, P, K); got shape {B.shape}")
    if offset is not None:
        off = np.asarray(offset, dtype=np.float64)
        off = off[:, None] if off.ndim == 1 else off
        eta = eta + (off[..., None] if eta.ndim == 3 else off)
    return eta


def term_contributions(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(N, T, K): each term's share of the log-odds. Sums over T to `linear_predictor(X, B)`.

    Pass the posterior (or prior) mean of B. This is where drivers come from -- no SHAP.
    """
    out = np.empty((X.shape[0], len(TERMS), K))
    for j, t in enumerate(TERMS):
        s = TERM_SLICE[t.name]
        out[:, j, :] = X[:, s] @ B[s, :]
    return out

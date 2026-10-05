"""The eval report contract: what `eval/report.py` assembles and `GET /report` serves."""

from __future__ import annotations

from pydantic import Field

from leeward.contracts.base import Base

# --------------------------------------------------------------------------- #
# GET /report
# --------------------------------------------------------------------------- #

class RecoveryRow(Base):
    parameter: str
    truth: float
    post_mean: float
    lo90: float
    hi90: float
    covered: bool


class CalibrationBin(Base):
    need: str
    predicted: float
    observed: float
    n: int


class FairnessRow(Base):
    stratum: str
    group: str
    #: Veteran-days in the group, then how many of them had a need, then how many of those
    #: got a call. Every rate below divides two of these, and a ratio shown without its
    #: denominator invites a reader to trust a number built on four events.
    n: int
    n_events: int = 0
    n_called: int | None = None
    ece: float
    #: The raw rate stays in the contract. It is the honest denominator, and showing only
    #: the friendlier number below would be the suppression the audit exists to prevent.
    fnr: float
    fnr_ratio_to_cohort: float
    #: 1 - fnr, and its ratio: the same measurement with the ceiling taken off. On a cohort
    #: whose pooled FNR is near 1 this is the only one of the two that can move.
    reach: float = 0.0
    reach_ratio_to_cohort: float = 1.0
    #: Share of the group's events that got one of the scarce daily calls, and its ratio.
    #: **None means not measured, not nobody** -- an audit run without a call list.
    coverage: float | None = None
    coverage_ratio_to_cohort: float | None = None
    direction: str = Field(
        default="on_par",
        description="reached_more | reached_less | on_par -- which side of the same 20 "
                    "percent bar the group's reach falls on. reached_more is a result, "
                    "not a pass")
    flagged: bool = Field(description="True when relative FNR gap exceeds 20 percent")


class DiscriminationRow(Base):
    """Per need, on the held-out window. TRIPOD+AI's first leg; `leeward/eval/discrimination.py`.

    `within_day_auc` is the headline and `pooled_auc` is the flattering one: the call list is
    chosen within a day, so pooled AUC includes credit for knowing today is a heat wave. The
    UI shows both, always, and never the pooled one alone.

    The AUCs and `pr_auc` are null when the window has no events to rank -- "we could not
    measure this" must not render as "the model scored zero".
    """
    need: str
    within_day_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    pooled_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    pr_auc: float | None = Field(default=None, ge=0.0, le=1.0)
    #: 1 - Brier/Brier_null, the Brier skill score. **The score on which the constant does
    #: not win**: it is 0.0 for a constant at the base rate and 1.0 for a perfect predictor,
    #: and unlike ECE it is strictly proper, so it keeps the sharpness term. Negative is
    #: allowed and means worse than knowing nothing, so there is no lower bound here.
    scaled_brier: float | None = Field(default=None, le=1.0)
    #: The raw Brier. Prevalence-dependent, and shipped only so `scaled_brier` is not read
    #: as it: predicting zero for everyone scores 0.0034 at a 0.34% base rate.
    brier: float | None = Field(default=None, ge=0.0)
    lift_at_1pct: float | None = None
    lift_at_10pct: float | None = None
    #: min(1/q, 1/base_rate): the best lift@1% any predictor could score here. Lift is
    #: uninterpretable without it -- 10.7x of a possible 42.9x is not 10.7x of a possible 11.
    lift_ceiling_1pct: float | None = None
    base_rate: float
    n: int
    n_events: int
    #: Held-out days on which the need happened at all, so had an AUC to contribute, and
    #: days there were. Eventless days are dropped, which conditions on the outcome, so the
    #: denominator travels with the numerator rather than being left off the slide.
    n_days: int
    n_days_total: int = 0


class AblationRow(Base):
    dropped: str
    ece: float
    harm_averted_at_40: float


class DecisionQualityRow(Base):
    k: int
    strategy: str
    harm_averted: float


class ReportResponse(Base):
    model_rung: int = Field(ge=0, le=3)
    rhat_max: float | None = None
    divergences: int | None = None
    recovery: list[RecoveryRow] = Field(default_factory=list)
    recovery_coverage: float | None = None
    calibration: list[CalibrationBin] = Field(default_factory=list)
    ece_by_need: dict[str, float] = Field(default_factory=dict)
    #: The control for `ece_by_need`: what a single number equal to each need's base rate
    #: scores on the same equal-mass bins. It is 0.0 by construction, which is *better* than
    #: the model's. Shown beside `ece_by_need`, never instead of it -- at these base rates
    #: calibration cannot separate the model from this, and `discrimination` is what can.
    constant_ece: dict[str, float] = Field(default_factory=dict)
    discrimination: list[DiscriminationRow] = Field(default_factory=list)
    ablations: list[AblationRow] = Field(default_factory=list)
    decision_quality: list[DecisionQualityRow] = Field(default_factory=list)
    fairness: list[FairnessRow] = Field(default_factory=list)
    #: True when any fairness row is flagged. The UI shows it either way, never hides it.
    fairness_failed: bool = False
    generated_at: str | None = None

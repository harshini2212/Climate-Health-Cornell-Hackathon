"""API request and response models. The contract between the `api` lane and the `ui` lane.

The UI is built against these before any real scores exist, so the shapes here are frozen
the moment they are pushed. Adding an optional field is fine; renaming or removing one is
a contract change and belongs in docs/SPEC.md first.
"""

from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, ConfigDict, Field

from leeward.schema import ACTIONS, DEFAULT_CAPACITY, NEEDS, TIERS


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- #
# GET /forecast?scenario=&day=
# --------------------------------------------------------------------------- #

class ZipHazard(Base):
    modzcta: str
    date: Date
    heat_index_max_f: float
    hot_day: bool
    heat_alert: bool
    pm25: float
    smoke_alert: bool
    flood_warning: bool
    flash_flood_emergency: bool
    surge_ft: float
    evac_zone_ordered: int
    outage_frac: float
    mail_delivery_disrupted: bool


class FacilityStatus(Base):
    facility_id: str
    #: One row per facility **per day**, like `ZipHazard`. A closure is a fact about a day,
    #: not about the window: Sandy shuts station 630 for 45 days and the ribbon has to put
    #: that on the day it starts. A caller that wants the window's answer takes `.any()`.
    date: Date
    name: str
    lat: float
    lon: float
    evac_zone: int
    site_down: bool
    site_dependent_services: bool


class ForecastResponse(Base):
    scenario: str
    day: int
    dates: list[Date]
    zips: list[ZipHazard]
    facilities: list[FacilityStatus]
    #: Free-text banner, e.g. "Coastal flood warning, zones 1-2, landfall in 3 days".
    headline: str | None = None


# --------------------------------------------------------------------------- #
# GET /scores?date=&need=
# --------------------------------------------------------------------------- #

class ZipScore(Base):
    modzcta: str
    need: str
    expected_count: float = Field(description="Sum of p_mean over the panel in this ZIP")
    lo80: float
    hi80: float
    n_panel: int


class ScoresResponse(Base):
    date: Date
    need: str
    zips: list[ZipScore]
    facilities: list[ZipScore] = Field(default_factory=list,
                                       description="Same shape, keyed by facility_id in modzcta")
    model_rung: int = Field(ge=0, le=3, description="Which ladder rung produced these")


# --------------------------------------------------------------------------- #
# GET /veteran/{id}?date=
# --------------------------------------------------------------------------- #

class NeedScore(Base):
    need: str
    p_mean: float
    p_lo80: float
    p_hi80: float
    p_epistemic_share: float
    #: What this veteran's number would be if the fields the VA does not have on file turned
    #: out to be their least- and most-risky values. Null when the record has no gap, which
    #: is not the same as a gap that would not move the number.
    p_gap_lo: float | None = None
    p_gap_hi: float | None = None
    drivers: list[str] = Field(default_factory=list, max_length=3)
    driver_contribs: list[float] = Field(default_factory=list, max_length=3)


class MedicationFlags(Base):
    """What the prescription list says, separate from what the diagnosis list says."""
    n_active_meds: int
    thermoreg_score: float
    acb_score: int
    combo_raas_diuretic: bool = Field(description="The combination CDC names explicitly")
    renal_triple: bool
    cold_chain: bool
    controlled: bool = Field(description="Retail emergency refill excludes these")
    narrow_ti: bool
    mail_order_pharmacy: bool
    days_supply_remaining: int
    #: Plain phrases for the card, e.g. "hydrochlorothiazide + lisinopril on a 96F day".
    notes: list[str] = Field(default_factory=list)


class VeteranCard(Base):
    veteran_id: str
    name_display: str
    age: int
    modzcta: str
    borough: str
    facility_id: str
    facility_name: str
    date: Date
    tier: str
    why_this_tier: str
    needs: list[NeedScore]
    medications: MedicationFlags
    conditions: list[str] = Field(default_factory=list)
    powered_equipment: str
    caregiver: str
    floor: str
    evac_zone: int
    planned_actions: list[str] = Field(default_factory=list)
    is_synthetic: bool = True


# --------------------------------------------------------------------------- #
# POST /actions
# --------------------------------------------------------------------------- #

class ActionsRequest(Base):
    #: The **do-by day**: the day this work list is worked. An action here may be for a risk
    #: that lands later -- see `ActionRow.lead_days`.
    date: Date
    capacity: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_CAPACITY))
    group_floor: dict[str, float] | None = Field(
        default=None, description="Minimum share of slots per group, e.g. {'borough': 0.1}")
    prior_scale: float = Field(default=1.0, description="0.5, 1.0 or 2.0; picks a cached posterior")
    scenario: str = "sandy_then_heat"
    limit: int | None = Field(
        default=None, ge=0,
        description="Return at most this many rows, highest EHA first. The allocation is "
                    "unchanged and every count still describes all of it, so a board that "
                    "renders a top slice can say '40 of 6,303'. Null means every row.")


class ActionRow(Base):
    action_id: str
    rank: int
    veteran_id: str
    name_display: str
    modzcta: str
    borough: str
    action: str
    tier: str
    eha: float
    capacity_bucket: str
    owner: str
    lead_days: int = Field(
        ge=0, description="Days ahead of the risk this has to happen to work at all. The "
                          "response's `date` is the do-by day, so the risk day this action "
                          "is for is `date + lead_days`. Zero means it still works today.")
    headline: str = Field(
        description="Card-safe reason line: urgency and timing, no condition, medicine or "
                    "service. The only one of these three a wall-mounted board may render.")
    rationale: str = Field(
        description="The full sentence, naming the service and the driver. Drill-down only.")
    top_driver: str | None = Field(
        default=None, description="Names a condition or a medicine. Drill-down only.")
    message_id: str | None = None


class BaselineResult(Base):
    name: str = Field(description="leeward | rank_by_age | rank_by_chronic | random")
    total_eha: float


class ActionsResponse(Base):
    #: The do-by day this list is worked on. Every row's risk day is `date + lead_days`.
    date: Date
    capacity: dict[str, int]
    #: The allocation, highest EHA first, cut to `ActionsRequest.limit` if one was asked for.
    #: Every count below is of the whole allocation, never of this list.
    actions: list[ActionRow]
    total_eha: float
    baselines: list[BaselineResult] = Field(default_factory=list)
    n_panel: int
    n_selected: int
    counts_by_tier: dict[str, int] = Field(default_factory=dict)
    n_not_reached: int = Field(
        default=0, ge=0,
        description="Veterans this capacity does not reach with a person at all: Act-now or "
                    "Find-out, offered a human action by an uncapped team, and given none "
                    "here. A free verified text is not being reached. It is the same number "
                    "a second uncapped request would have shown, computed in the one pass "
                    "that already values every candidate, so nobody has to ask twice.")
    n_too_late: int = Field(
        default=0, ge=0,
        description="Actions for the risk days ahead whose do-by day already falls before "
                    "`date`, so this work day cannot offer them however much capacity it "
                    "has. On the first day of a forecast that is exactly what arriving late "
                    "cost; on a later day it is the work that had to happen before today. "
                    "Show the number either way.")
    model_rung: int = Field(ge=0, le=3)


# --------------------------------------------------------------------------- #
# GET /message/{action_id}
# --------------------------------------------------------------------------- #

class Message(Base):
    message_id: str
    action_id: str
    veteran_id: str
    channel: str = Field(description="VEText | MHV | care_team_phone")
    addressed_to: str = Field(description="veteran | caregiver")
    verification_phrase: str = Field(description="Four words the veteran can read back")
    body: str
    #: Mandatory elements, surfaced separately so the UI can show them as a checklist
    # and `test_guardrails.py` can assert they are present.
    includes_never_pay_line: bool
    includes_vsafe: bool
    includes_crisis_line: bool
    scam_card_url: str | None = None


# --------------------------------------------------------------------------- #
# POST /log
# --------------------------------------------------------------------------- #

class LogRequest(Base):
    action_id: str
    veteran_id: str
    date: Date
    done: bool
    reached: bool
    need_occurred: bool | None = None
    partner_ack: bool | None = None
    logged_by: str


class LogResponse(Base):
    ok: bool
    n_rows: int


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
    n: int
    ece: float
    fnr: float
    fnr_ratio_to_cohort: float
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


__all__ = [
    "NEEDS", "TIERS", "ACTIONS", "DEFAULT_CAPACITY",
    "ForecastResponse", "ZipHazard", "FacilityStatus",
    "ScoresResponse", "ZipScore",
    "VeteranCard", "NeedScore", "MedicationFlags",
    "ActionsRequest", "ActionsResponse", "ActionRow", "BaselineResult",
    "Message", "LogRequest", "LogResponse",
    "ReportResponse", "RecoveryRow", "CalibrationBin", "FairnessRow",
    "AblationRow", "DecisionQualityRow", "DiscriminationRow",
]

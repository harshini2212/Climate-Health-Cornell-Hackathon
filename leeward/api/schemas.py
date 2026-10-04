"""API request and response models. The contract between the API and the UI.

The UI is built against these before any real scores exist, so the shapes here are frozen
the moment they are pushed. Adding an optional field is fine; renaming or removing one is
a contract change and belongs in docs/SPEC.md first.
"""

from __future__ import annotations

from datetime import date as Date

from pydantic import Field

# Models another layer also produces live in `leeward/contracts/`; they are re-exported here
# unchanged, so this module is still the whole API contract the UI is typed against.
from leeward.contracts import (
    AblationRow,
    Base,
    CalibrationBin,
    DecisionQualityRow,
    DiscriminationRow,
    FairnessRow,
    Message,
    RecoveryRow,
    ReportResponse,
)
from leeward.schema import ACTIONS, DEFAULT_CAPACITY, NEEDS, TIERS

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

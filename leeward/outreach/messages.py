"""Outreach messages (SPEC §8): `render(action, veteran) -> Message`.

After a disaster, scammers pose as the agency that is helping. So every message Leeward
sends has to be distinguishable from a scam, and that is built into the shape of `render`
rather than left to each template: a template writes only the *middle* of a message, and
`render` wraps it in the parts a veteran can check -- the channel tag, the four-word
phrase, the never-pay line, VSAFE, and the Crisis Line. No template can leave one out.

Three rules the spec sets, and where they live:

* **Caregiver messages** (SPEC §7.4b). With a caregiver and the veteran's consent to contact
  them, `care_team_call` and `verified_text` go to the caregiver first: they name the
  veteran, speak to the caregiver as "you", and say how this caregiver can act.
* **Low-assets messages** (`low_assets` comes from CDC PLACES `shututility`). They say what
  is free and never suggest anything that costs money; rides are *booked*, not offered.
* **No diagnosis, anywhere.** A text sits on a lock screen. Templates never name a
  condition, a drug or a piece of equipment, only "your medicines" and "medical equipment
  that plugs in".

Leeward never changes a medication. Medication actions flag a veteran for the VA pharmacist,
and the messages say so: keep taking everything as prescribed until the pharmacist decides.

Nothing here links anywhere. A veteran told to distrust links is never sent one, so
`Message.scam_card_url` stays None until a verified card exists to point at.

The "free" lines mirror SPEC §8 (HEAP, cooling centers, emergency refill voucher, VA
transport). They are the spec's claims, not yet sourced in docs/sources.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from leeward.api.schemas import Message
from leeward.outreach import verify
from leeward.schema import ACTIONS, CAREGIVER, TIERS

NEVER_PAY = "The VA will never ask you to pay, wire money, or share bank details."
VSAFE = "VSAFE 833-388-7233"
CRISIS_LINE = "Veterans Crisis Line: dial 988, press 1"

#: SPEC §7.4b: with a consenting caregiver, these two go to the caregiver first.
CAREGIVER_FIRST = ("care_team_call", "verified_text")

#: Which VA channel carries which kind of action. Refills and pharmacist contact are My
#: HealtheVet's job, appointments and notices are VEText's, and anything that needs a live
#: conversation is the care team's own phone.
CHANNEL_FOR = {
    "care_team_call": "care_team_phone",
    "check_in_call": "care_team_phone",
    "backup_power_plan": "care_team_phone",
    "cold_chain_plan": "care_team_phone",
    "evacuation_assist": "care_team_phone",
    "early_refill": "MHV",
    "switch_to_local_pickup": "MHV",
    "pharmacist_med_review": "MHV",
    "controlled_substance_bridge": "MHV",
    "verified_text": "VEText",
    "alt_site_booking": "VEText",
    "cooling_center_ride": "VEText",
    "clean_air_room": "VEText",
    "heap_application": "VEText",
    "assign_buddy": "VEText",
}

#: One line per tier, so the tone tracks urgency without changing what is promised.
LEAD = {
    "act_now": "This is time-sensitive, so please read it today.",
    "find_out": "Your care team wants to make sure you are set.",
    "self_serve": "This is a heads-up so you can plan ahead.",
    "everyday": "A note from your care team.",
}

#: What this caregiver can do about it, by the kind of caregiver they are.
_REACH = {
    "informal_coresident": "Please stay with {name}, or check on them in person, until we reach you.",
    "informal_remote": "Please phone {name} today to see how they are and whether they have "
                       "what they need.",
    "va_pcafc": "Please check on {name} today.",
}


@dataclass(frozen=True)
class _Ctx:
    """Everything a template may branch on. Nothing here is a diagnosis."""
    name: str
    to_caregiver: bool
    reach: str
    low_assets: bool
    #: SPEC §7.4b: a low-assets veteran, or one with a transport barrier, gets the ride
    #: booked rather than offered.
    booked: bool
    home_ac: bool
    basement: bool


# --------------------------------------------------------------------------- #
# Templates: the middle of a message. Each returns lines; `render` does the rest.
# --------------------------------------------------------------------------- #

def _care_team_call(c: _Ctx) -> list[str]:
    if c.to_caregiver:
        return [f"A member of the VA care team will call you today about {c.name}, to go over "
                "the plan for the next few days.", c.reach]
    return ["A member of your VA care team will call you today to go over your plan and make "
            "sure you have what you need."]


def _verified_text(c: _Ctx) -> list[str]:
    if c.to_caregiver:
        return [f"Weather that could affect {c.name}'s health is expected near them.",
                f"Please help {c.name} make sure they have their medicines, a charged phone, "
                "and any medical equipment that plugs in.", c.reach]
    return ["Weather that could affect your health is expected near you.",
            "Please make sure you have your medicines, a charged phone, and any medical "
            "equipment that plugs in. Tell your care team if your plans change or you need help."]


def _check_in_call(c: _Ctx) -> list[str]:
    return ["A member of your VA care team will call you for a short check-in. We want to hear "
            "how you are doing and whether you have what you need. There is nothing to prepare."]


def _backup_power_plan(c: _Ctx) -> list[str]:
    return ["Your care team will call you to make a plan for a power outage: backup power for "
            "any medical equipment you rely on, where to go, and who to call. Please keep your "
            "phone charged."]


def _cold_chain_plan(c: _Ctx) -> list[str]:
    return ["Some of your medicines need to stay cold. Your care team will call you to plan for "
            "a power outage so they stay safe. Until we talk, keep the refrigerator door closed "
            "as much as you can."]


def _early_refill(c: _Ctx) -> list[str]:
    return ["Your care team is arranging an early refill of your current prescriptions so you "
            "do not run short if deliveries are delayed. Your medicines stay exactly the same. "
            "Your VA pharmacy will confirm how and when you will get them."]


def _switch_to_local_pickup(c: _Ctx) -> list[str]:
    return ["Mail delivery to your area may be delayed. Your care team can arrange for you to "
            "pick up your refill at a VA pharmacy instead of waiting for the mail. Your "
            "medicines stay the same."]


def _pharmacist_med_review(c: _Ctx) -> list[str]:
    return ["A VA clinical pharmacist would like to talk with you about taking your medicines "
            "safely in extreme weather. Until you have spoken with them, keep taking every "
            "medicine exactly as prescribed. Only the pharmacist or your doctor can change "
            "anything; this message changes nothing."]


def _controlled_substance_bridge(c: _Ctx) -> list[str]:
    return ["Your care team is arranging for the VA to fill one of your prescriptions directly, "
            "ahead of the weather. The VA pharmacy will contact you with the details. Your "
            "prescription is not changing."]


def _alt_site_booking(c: _Ctx) -> list[str]:
    return ["Your usual VA site may be closed. Your care team is booking your regular "
            "appointment at another VA site so you do not miss it. We will call to confirm the "
            "place and time. If you cannot get there, tell us."]


def _cooling_center_ride(c: _Ctx) -> list[str]:
    lines = ["Very hot weather is expected near you."]
    lines.append(
        "Your care team has booked a ride for you to a cooling center. We will call to "
        "confirm your pickup time." if c.booked else
        "Your care team will call to offer you a ride to a cooling center.")
    if c.home_ac and not c.low_assets:     # a veteran who cannot fund AC is never told to run it
        lines.append("Please use your air conditioning at home and stay in the coolest room.")
    return lines


def _clean_air_room(c: _Ctx) -> list[str]:
    return ["Smoke is making the air unhealthy near you. Please stay indoors and keep your "
            "windows closed.",
            "Your care team has booked a ride for you to a place with clean indoor air. We will "
            "call to confirm your pickup time." if c.booked else
            "Your care team will call to offer you a ride to a place with clean indoor air."]


def _evacuation_assist(c: _Ctx) -> list[str]:
    """Flood messages add an evacuation center with step-free access and a pack list (SPEC §8)."""
    lines = ["An evacuation order may apply to your area.",
             "Your care team has booked transport for you to an evacuation center with "
             "step-free access. We will call to confirm your pickup time." if c.booked else
             "Your care team will call to arrange your transport to an evacuation center with "
             "step-free access.",
             "Pack your medicines, any medical equipment, chargers, and ID."]
    if c.basement:
        lines.append("Do not stay in a basement: move to a higher floor now.")
    lines.append("If water is rising or you are in danger, call 911.")
    return lines


def _heap_application(c: _Ctx) -> list[str]:
    return ["You may qualify for HEAP, New York's Home Energy Assistance Program. A member of "
            "your care team will help you fill in the application."]


def _assign_buddy(c: _Ctx) -> list[str]:
    return ["Your care team is setting up a check-in partner: a trained volunteer who can look "
            "in on you during this weather. We will tell you who to expect before anyone "
            "contacts you."]


TEMPLATES: dict[str, Callable[[_Ctx], list[str]]] = {
    "care_team_call": _care_team_call,
    "verified_text": _verified_text,
    "check_in_call": _check_in_call,
    "backup_power_plan": _backup_power_plan,
    "cold_chain_plan": _cold_chain_plan,
    "early_refill": _early_refill,
    "switch_to_local_pickup": _switch_to_local_pickup,
    "pharmacist_med_review": _pharmacist_med_review,
    "controlled_substance_bridge": _controlled_substance_bridge,
    "alt_site_booking": _alt_site_booking,
    "cooling_center_ride": _cooling_center_ride,
    "clean_air_room": _clean_air_room,
    "evacuation_assist": _evacuation_assist,
    "heap_application": _heap_application,
    "assign_buddy": _assign_buddy,
}

#: What is free, per action, for a `low_assets` veteran (SPEC §8). Anything not listed gets
#: the generic line: a low-assets message always says what is free.
FREE = {
    "cooling_center_ride": "Free for you: the ride and the cooling center.",
    "clean_air_room": "Free for you: the ride and the clean-air space.",
    "evacuation_assist": "Free for you: the ride and the evacuation center.",
    "heap_application": "Free for you: applying for HEAP, and help filling in the application.",
    "early_refill": "Free for you: the emergency refill voucher.",
    "switch_to_local_pickup": "Free for you: VA transport to pick up your refill.",
    "alt_site_booking": "Free for you: your appointment and VA transport to it.",
}
FREE_DEFAULT = "Free for you: this help from your VA care team."


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #

def _need(row: Mapping[str, Any], key: str, what: str) -> Any:
    value = row.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{what} has no {key!r}; cannot render a message without it")
    return value


def _mapping(obj: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise TypeError(f"{what} must be a mapping (a table row as a dict), got {type(obj).__name__}")
    return obj


def render(action: Mapping[str, Any], veteran: Mapping[str, Any]) -> Message:
    """The message for one row of `actions.parquet`, about one row of `cohort.parquet`.

    Both arguments are plain dicts (`DataFrame.to_dicts()` rows). Raises ValueError rather
    than guessing when something needed is missing or outside the contract's vocabulary:
    a wrong message to a veteran is worse than an error on the care team's screen.
    """
    action = _mapping(action, "action")
    veteran = _mapping(veteran, "veteran")

    kind = str(_need(action, "action", "action row"))
    if kind not in TEMPLATES:
        raise ValueError(f"no message template for action {kind!r}; actions are {ACTIONS}")
    tier = str(_need(action, "tier", "action row"))
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; tiers are {TIERS}")
    action_id = str(_need(action, "action_id", "action row"))
    veteran_id = str(_need(action, "veteran_id", "action row"))
    day = _need(action, "date", "action row")

    name = str(_need(veteran, "name_display", "veteran row")).strip()
    caregiver = str(veteran.get("caregiver") or "none")
    if caregiver not in CAREGIVER:
        raise ValueError(f"unknown caregiver {caregiver!r}; values are {CAREGIVER}")

    to_caregiver = (kind in CAREGIVER_FIRST and caregiver != "none"
                    and bool(veteran.get("caregiver_contact_consent")))
    low_assets = bool(veteran.get("low_assets"))
    ctx = _Ctx(
        name=name,
        to_caregiver=to_caregiver,
        reach=_REACH.get(caregiver, "").format(name=name),
        low_assets=low_assets,
        booked=low_assets or bool(veteran.get("transport_barrier")),
        home_ac=bool(veteran.get("home_ac")),
        basement=veteran.get("floor") == "basement",
    )

    channel = CHANNEL_FOR[kind]
    phrase = verify.phrase(veteran_id, day)

    if to_caregiver:
        greeting = (f"Hello. This is the VA care team writing about {name}. You are listed as "
                    "their support person, and we have their permission to contact you.")
    else:
        greeting = f"Hello {name},"

    lines = [
        f"[{channel}] VA Care Team",
        greeting,
        LEAD[tier],
        *TEMPLATES[kind](ctx),
        *([FREE.get(kind, FREE_DEFAULT)] if low_assets else []),
        f"Verification phrase: {phrase}.",
        "Real VA staff can say these four words to you, and you can read them back to confirm "
        "it is us. If someone contacting you cannot say them, do not continue.",
        NEVER_PAY,
        f"Suspect a scam? Call {VSAFE}.",
        f"{CRISIS_LINE}. In an emergency, call 911.",
    ]
    body = "\n".join(lines)

    message_id = action.get("message_id") or f"msg-{action_id}"
    return Message(
        message_id=str(message_id),
        action_id=action_id,
        veteran_id=veteran_id,
        channel=channel,
        addressed_to="caregiver" if to_caregiver else "veteran",
        verification_phrase=phrase,
        body=body,
        includes_never_pay_line=NEVER_PAY in body,
        includes_vsafe=VSAFE in body,
        includes_crisis_line=CRISIS_LINE in body,
        scam_card_url=None,
    )

"""Outreach messages (SPEC §8): the promises made to a veteran, so hard failures.

`test_guardrails.py` checks the three strings every message must carry against the real
cohort. This file checks everything else the spec promises, against small hand-built
veterans so each rule is tested by the case that triggers it:

  * every action in the vocabulary renders, and every render is distinguishable from a scam
  * caregiver messages name the veteran, speak to the caregiver, and carry no diagnosis
  * low-assets messages say what is free and never suggest anything that costs money
  * Leeward never changes a medication, and the message says so
"""

from __future__ import annotations

import re
from datetime import date

import pytest

from leeward.api.schemas import Message
from leeward.outreach import verify
from leeward.outreach.messages import render
from leeward.schema import ACTIONS, CHANNELS, MANDATORY_MESSAGE_ELEMENTS

DAY = date(2026, 7, 16)

NEVER_PAY = "The VA will never ask you to pay, wire money, or share bank details"
VSAFE = "VSAFE 833-388-7233"
CRISIS = "Veterans Crisis Line: dial 988, press 1"

#: Anything a text on a lock screen must not reveal, whoever it is addressed to.
DIAGNOSIS_WORDS = (
    "copd", "asthma", "chf", "heart failure", "diabet", "dialysis", "kidney", "ckd", "cancer",
    "chemo", "ptsd", "depress", "anxiety", "hypertension", "oxygen", "ventilator", "insulin",
    "methadone", "opioid", "narcotic", "controlled substance",
)

#: Ways a message could nudge a veteran who cannot self-fund anything into spending.
PAID_OPTIONS = re.compile(
    r"\b(hotel|motel|taxi|uber|lyft|rideshare|buy|purchase|rent|fee|copay|cost|charge|"
    r"air conditioning|air conditioner|fan)\b|\$", re.I)

#: Medication instructions Leeward must never give: it flags, the pharmacist decides.
MED_CHANGES = ("stop taking", "skip your", "change your dose", "change your medic",
               "adjust your", "switch your medic", "double your", "reduce your", "increase your")


def _vet(**over) -> dict:
    base = {
        "veteran_id": "SYN-000123", "name_display": "Marcus O'Brien",
        "caregiver": "none", "caregiver_contact_consent": False,
        "low_assets": False, "transport_barrier": False, "home_ac": True, "floor": "upper",
        # What a lock-screen text must never leak, present so a leak would be caught.
        "copd": True, "ptsd": True, "ckd_dialysis": True, "diabetes": True,
        "powered_equipment": "oxygen", "med_cold_chain": True, "med_controlled": True,
        "on_methadone_otp": True, "conditions": ["13645005", "40055000"],
    }
    return base | over


def _action(action: str, tier: str = "act_now", **over) -> dict:
    base = {
        "action_id": f"a-{action}", "date": DAY, "veteran_id": "SYN-000123", "action": action,
        "tier": tier, "eha": 1.0, "rank": 1, "capacity_bucket": "call",
        "rationale": "Pharmacist review before the heat", "owner": "care_team",
        "message_id": None,
    }
    return base | over


def _body(action: str, tier: str = "act_now", **vet) -> str:
    return render(_action(action, tier), _vet(**vet)).body


PROFILES = {
    "plain": {},
    "low_assets": {"low_assets": True},
    "transport_barrier": {"transport_barrier": True},
    "basement": {"floor": "basement"},
    "coresident": {"caregiver": "informal_coresident", "caregiver_contact_consent": True},
    "remote": {"caregiver": "informal_remote", "caregiver_contact_consent": True},
    "pcafc_low_assets": {"caregiver": "va_pcafc", "caregiver_contact_consent": True,
                         "low_assets": True},
}
EVERY_MESSAGE = [pytest.param(a, p, id=f"{a}-{p}") for a in ACTIONS for p in PROFILES]


# --------------------------------------------------------------------------- #
# Every message is distinguishable from a scam
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(("action", "profile"), EVERY_MESSAGE)
def test_every_message_carries_the_anti_scam_elements(action: str, profile: str) -> None:
    msg = render(_action(action), _vet(**PROFILES[profile]))
    assert isinstance(msg, Message)
    for element in (*MANDATORY_MESSAGE_ELEMENTS, NEVER_PAY, VSAFE, CRISIS):
        assert element in msg.body, f"{action}/{profile}: missing {element!r}"
    assert msg.includes_never_pay_line and msg.includes_vsafe and msg.includes_crisis_line
    assert f"[{msg.channel}]" in msg.body, "the channel tag is part of the message, not metadata"
    assert msg.channel in CHANNELS
    assert msg.verification_phrase in msg.body
    assert len(msg.verification_phrase.split()) == 4


@pytest.mark.parametrize(("action", "profile"), EVERY_MESSAGE)
def test_no_links_and_no_phone_number_outside_the_allow_list(action: str, profile: str) -> None:
    body = render(_action(action), _vet(**PROFILES[profile])).body
    assert not re.search(r"https?://|www\.|\.(com|gov|org|net|io|ly)\b", body, re.I), (
        "no links at all: a veteran told to distrust links must never be sent one")
    numbers = set(re.findall(r"\d[\d-]*\d|\d", body))
    assert numbers <= {"833-388-7233", "988", "1", "911"}, f"unapproved digits: {numbers}"


@pytest.mark.parametrize(("action", "profile"), EVERY_MESSAGE)
def test_no_message_leaks_a_diagnosis_or_guesses_a_pronoun(action: str, profile: str) -> None:
    body = render(_action(action), _vet(**PROFILES[profile])).body.lower()
    for word in DIAGNOSIS_WORDS:
        assert word not in body, f"{action}/{profile}: {word!r} on a lock screen"
    assert not re.search(r"\b(he|she|him|his|hers?)\b", body), (
        "pronouns are never inferred from a name; use the name or they/their")


def test_phrase_is_the_veteran_days_phrase_and_shared_across_the_days_messages() -> None:
    want = verify.phrase("SYN-000123", DAY)
    for action in ACTIONS:
        assert render(_action(action), _vet()).verification_phrase == want
    other_day = render(_action("care_team_call", date=date(2026, 7, 17)), _vet())
    assert other_day.verification_phrase != want


def test_render_is_deterministic() -> None:
    assert render(_action("early_refill"), _vet()) == render(_action("early_refill"), _vet())


def test_identifiers_and_message_id() -> None:
    msg = render(_action("early_refill", action_id="abc123"), _vet())
    assert (msg.action_id, msg.veteran_id, msg.message_id) == ("abc123", "SYN-000123", "msg-abc123")
    assert msg.scam_card_url is None, "no verified scam-card link exists yet; do not invent one"
    given = render(_action("early_refill", action_id="abc123", message_id="msg-x"), _vet())
    assert given.message_id == "msg-x"


def test_channel_follows_the_kind_of_action() -> None:
    assert render(_action("care_team_call"), _vet()).channel == "care_team_phone"
    assert render(_action("early_refill"), _vet()).channel == "MHV"
    assert render(_action("verified_text"), _vet()).channel == "VEText"


def test_tier_changes_the_tone_but_not_the_promises() -> None:
    urgent, calm = _body("early_refill", "act_now"), _body("early_refill", "self_serve")
    assert urgent != calm
    for body in (urgent, calm):
        assert NEVER_PAY in body and VSAFE in body and CRISIS in body


def test_bad_input_fails_loudly() -> None:
    with pytest.raises(ValueError):
        render(_action("teleport_home"), _vet())
    with pytest.raises(ValueError):
        render({k: v for k, v in _action("care_team_call").items() if k != "date"}, _vet())
    with pytest.raises(ValueError):
        render(_action("care_team_call"), _vet(name_display=""))


# --------------------------------------------------------------------------- #
# Caregiver-addressed messages (SPEC §7.4b, §8)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("action", ["care_team_call", "verified_text"])
@pytest.mark.parametrize("caregiver", ["informal_coresident", "informal_remote", "va_pcafc"])
def test_caregiver_gets_the_message_when_the_veteran_consented(action: str, caregiver: str) -> None:
    msg = render(_action(action), _vet(caregiver=caregiver, caregiver_contact_consent=True))
    assert msg.addressed_to == "caregiver"
    assert "Marcus O'Brien" in msg.body, "a caregiver message must say who it is about"
    assert re.search(r"\byou\b", msg.body, re.I), "second person, to the caregiver"
    assert "Hello Marcus O'Brien" not in msg.body, "do not greet the veteran in a caregiver text"


@pytest.mark.parametrize("action", ["care_team_call", "verified_text"])
def test_caregiver_is_never_messaged_without_consent(action: str) -> None:
    msg = render(_action(action), _vet(caregiver="informal_coresident",
                                       caregiver_contact_consent=False))
    assert msg.addressed_to == "veteran"


def test_no_caregiver_means_the_veteran_gets_it() -> None:
    assert render(_action("verified_text"), _vet(caregiver="none")).addressed_to == "veteran"


@pytest.mark.parametrize("action", [a for a in ACTIONS if a not in ("care_team_call", "verified_text")])
def test_only_the_call_and_the_text_are_routed_to_a_caregiver(action: str) -> None:
    """SPEC §7.4b names exactly these two; a ride or a refill still goes to the veteran."""
    msg = render(_action(action), _vet(caregiver="informal_coresident",
                                       caregiver_contact_consent=True))
    assert msg.addressed_to == "veteran"


def test_caregiver_message_says_how_this_caregiver_can_act() -> None:
    cores = _body("care_team_call", caregiver="informal_coresident", caregiver_contact_consent=True)
    remote = _body("care_team_call", caregiver="informal_remote", caregiver_contact_consent=True)
    assert cores != remote
    assert "in person" in cores and "phone" in remote


# --------------------------------------------------------------------------- #
# Low-assets messages state what is free and never suggest a paid option
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("action", ACTIONS)
def test_low_assets_messages_say_what_is_free_and_suggest_nothing_paid(action: str) -> None:
    body = _body(action, low_assets=True)
    assert re.search(r"\bfree\b", body, re.I), f"{action}: a low-assets message must say what is free"
    # The mandatory never-pay line contains the word "pay" by design; judge everything else.
    rest = body.replace(NEVER_PAY, "")
    hit = PAID_OPTIONS.search(rest)
    assert hit is None, f"{action}: {hit.group(0)!r} suggests spending money"


def test_low_assets_messages_name_the_free_things_the_spec_lists() -> None:
    assert re.search(r"free.*cooling center", _body("cooling_center_ride", low_assets=True), re.I)
    assert "HEAP" in _body("heap_application", low_assets=True)
    assert "refill voucher" in _body("early_refill", low_assets=True)
    assert "VA transport" in _body("switch_to_local_pickup", low_assets=True)


def test_low_assets_rides_are_booked_not_suggested() -> None:
    for action in ("cooling_center_ride", "evacuation_assist", "clean_air_room"):
        body = _body(action, low_assets=True)
        assert "booked" in body, f"{action}: SPEC §7.4b says booked, not suggested"


def test_a_transport_barrier_also_gets_the_ride_booked() -> None:
    assert "booked" in _body("cooling_center_ride", transport_barrier=True)


def test_ride_is_offered_when_there_is_no_barrier() -> None:
    body = _body("cooling_center_ride")
    assert "booked" not in body and "offer" in body


def test_only_a_veteran_who_can_pay_for_it_is_told_to_use_their_air_conditioning() -> None:
    assert "air conditioning" in _body("cooling_center_ride", home_ac=True, low_assets=False)
    assert "air conditioning" not in _body("cooling_center_ride", home_ac=True, low_assets=True)
    assert "air conditioning" not in _body("cooling_center_ride", home_ac=False)


# --------------------------------------------------------------------------- #
# Content the spec asks for by name
# --------------------------------------------------------------------------- #

def test_evacuation_message_has_step_free_access_and_a_pack_list() -> None:
    body = _body("evacuation_assist")
    assert "step-free" in body
    for item in ("medicines", "equipment", "chargers", "ID"):
        assert item in body, f"pack list is missing {item!r}"


def test_a_basement_veteran_is_told_to_leave_the_basement_in_a_flood() -> None:
    assert "basement" in _body("evacuation_assist", floor="basement")
    assert "basement" not in _body("evacuation_assist", floor="upper")


# --------------------------------------------------------------------------- #
# Leeward never changes a medication
# --------------------------------------------------------------------------- #

MED_ACTIONS = ["early_refill", "switch_to_local_pickup", "pharmacist_med_review",
               "controlled_substance_bridge", "cold_chain_plan"]


@pytest.mark.parametrize("action", MED_ACTIONS)
def test_medication_messages_never_tell_a_veteran_to_change_anything(action: str) -> None:
    body = _body(action).lower()
    for phrase in MED_CHANGES:
        assert phrase not in body, f"{action}: {phrase!r}; only the pharmacist decides"


def test_pharmacist_review_says_who_decides() -> None:
    body = _body("pharmacist_med_review")
    assert "pharmacist" in body
    assert "keep taking" in body.lower(), "the one instruction that matters is: change nothing yet"

"""The partner sheet excludes a row it may not show; it never blanks a field in one.

Four veterans, each consenting to a different subset, checked by hand against
`leeward.outreach.export.CONSENT_FOR`. The guardrail in test_guardrails.py holds the one
rule that must never break; these pin the rest of what the module promises.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from leeward.outreach.export import COLUMNS, CONSENT_FOR, partner_sheet

DAY = date(2026, 7, 16)


def _cohort() -> pl.DataFrame:
    def vet(vid, partner, ride, housing):
        return dict(veteran_id=vid, name_display=f"Name {vid}", modzcta="10001",
                    borough="Manhattan", consent_partner_check=partner, consent_ride=ride,
                    consent_housing=housing)
    return pl.DataFrame([
        vet("ALL", True, True, True),       # consents to everything
        vet("RIDE", True, True, False),     # partner contact and rides, not housing
        vet("BARE", True, False, False),    # partner contact and nothing more
        vet("NONE", False, True, True),     # ride and housing but NOT partner contact
    ])


def _actions(*pairs: tuple[str, str]) -> pl.DataFrame:
    return pl.DataFrame([dict(date=DAY, veteran_id=v, action=a, tier="act_now", rank=i + 1,
                              eha=1.0, rationale="dialysis at a closed station")
                         for i, (v, a) in enumerate(pairs)])


def _rows(sheet: pl.DataFrame) -> set[tuple[str, str]]:
    return set(zip(sheet["veteran_id"], sheet["action"], strict=True))


def test_a_row_needs_partner_contact_and_the_consent_its_action_needs() -> None:
    sheet = partner_sheet(_actions(
        ("ALL", "cooling_center_ride"), ("ALL", "heap_application"), ("ALL", "assign_buddy"),
        ("RIDE", "cooling_center_ride"), ("RIDE", "heap_application"),
        ("RIDE", "assign_buddy"), ("BARE", "cooling_center_ride"), ("BARE", "assign_buddy"),
        ("NONE", "cooling_center_ride"), ("NONE", "assign_buddy"),
    ), _cohort())
    assert _rows(sheet) == {
        ("ALL", "cooling_center_ride"), ("ALL", "heap_application"), ("ALL", "assign_buddy"),
        ("RIDE", "cooling_center_ride"), ("RIDE", "assign_buddy"),
        ("BARE", "assign_buddy"),
    }


def test_every_ride_type_needs_ride_consent_and_heap_needs_housing() -> None:
    rides = [a for a, c in CONSENT_FOR.items() if c == "consent_ride"]
    assert set(rides) == {"cooling_center_ride", "clean_air_room", "evacuation_assist"}
    assert CONSENT_FOR["heap_application"] == "consent_housing"
    got = partner_sheet(_actions(*[("BARE", a) for a in rides]), _cohort())
    assert got.height == 0, "no ride consent, no ride row"


def test_actions_a_partner_never_performs_stay_inside_the_va() -> None:
    inside = ["care_team_call", "check_in_call", "early_refill", "pharmacist_med_review",
              "controlled_substance_bridge", "cold_chain_plan", "alt_site_booking",
              "backup_power_plan", "switch_to_local_pickup", "verified_text"]
    sheet = partner_sheet(_actions(*[("ALL", a) for a in inside]), _cohort())
    assert sheet.height == 0, "a fully consenting veteran's clinical actions must not be listed"


def test_the_sheet_carries_no_risk_or_diagnosis() -> None:
    sheet = partner_sheet(_actions(("ALL", "cooling_center_ride")), _cohort())
    assert sheet.columns == COLUMNS
    for leaked in ("tier", "rank", "eha", "rationale", "top_driver", "p_mean"):
        assert leaked not in sheet.columns
    assert sheet.row(0, named=True)["name_display"] == "Name ALL"


def test_excluded_means_absent_not_blanked() -> None:
    sheet = partner_sheet(_actions(("NONE", "assign_buddy"), ("ALL", "assign_buddy")), _cohort())
    assert sheet["veteran_id"].to_list() == ["ALL"]
    assert sheet.null_count().sum_horizontal().item() == 0


def test_nothing_qualifying_is_a_header_only_sheet() -> None:
    sheet = partner_sheet(_actions(("NONE", "assign_buddy")), _cohort())
    assert sheet.height == 0 and sheet.columns == COLUMNS
    assert sheet.write_csv().strip() == ",".join(COLUMNS)


def test_rows_come_out_in_a_stable_order() -> None:
    a = _actions(("RIDE", "assign_buddy"), ("ALL", "cooling_center_ride"), ("ALL", "assign_buddy"))
    got = partner_sheet(a, _cohort())
    assert _rows(got) == _rows(partner_sheet(a.reverse(), _cohort()))
    assert got["veteran_id"].to_list() == ["ALL", "ALL", "RIDE"]
    assert got.filter(pl.col("veteran_id") == "ALL")["action"].to_list() == [
        "assign_buddy", "cooling_center_ride"]

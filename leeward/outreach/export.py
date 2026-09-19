"""The partner sheet: who a community partner may be asked to help, and nothing more (SPEC §8).

    partner_sheet(actions, cohort) -> one row per (veteran, action) a partner would carry out

Two rules, and both exclude a row rather than blank a field in it. A redacted row still tells
a partner that the veteran is on a list.

1. The veteran consented to partner contact at all (`consent_partner_check`).
2. For an action that needs a further consent, the veteran gave that one too: rides and
   evacuation help need `consent_ride`; a HEAP application needs `consent_housing`.

Only the actions a partner performs are listed. Calls, refills, booking and everything the
pharmacist owns stay inside the VA, so no medication or clinical action can reach the sheet.
For the same reason it carries no tier, rank, EHA, rationale or driver: those are risk and
diagnosis in plain words. A partner gets who, where, what help, and the consents that allow it.

Which consent gates which action is this file's judgement, not the SPEC's; change `CONSENT_FOR`
to change it. The guardrail in `tests/test_guardrails.py` holds the line that matters.
"""

from __future__ import annotations

import polars as pl

#: action -> the consent it needs beyond `consent_partner_check`, or None for that one alone.
CONSENT_FOR: dict[str, str | None] = {
    "assign_buddy": None,
    "cooling_center_ride": "consent_ride",
    "clean_air_room": "consent_ride",
    "evacuation_assist": "consent_ride",
    "heap_application": "consent_housing",
}

CONSENTS = ["consent_partner_check", "consent_ride", "consent_housing"]
COLUMNS = ["date", "veteran_id", "name_display", "modzcta", "borough", "action", *CONSENTS]


def partner_sheet(actions: pl.DataFrame, cohort: pl.DataFrame) -> pl.DataFrame:
    """The partner sheet for the days in `actions`. Header only when nobody qualifies."""
    allowed = pl.lit(False)
    for action, consent in CONSENT_FOR.items():
        allowed = allowed | ((pl.col("action") == action)
                             & (pl.col(consent) if consent else pl.lit(True)))
    return (actions.filter(pl.col("action").is_in(list(CONSENT_FOR)))
                   .join(cohort.select("veteran_id", "name_display", "modzcta", "borough",
                                       *CONSENTS), on="veteran_id", how="inner")
                   .filter(pl.col("consent_partner_check") & allowed)
                   .select(COLUMNS)
                   .sort("date", "veteran_id", "action"))

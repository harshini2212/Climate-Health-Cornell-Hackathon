"""The outreach message contract: what `outreach/messages.py` renders and `GET /message` serves."""

from __future__ import annotations

from pydantic import Field

from leeward.contracts.base import Base

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

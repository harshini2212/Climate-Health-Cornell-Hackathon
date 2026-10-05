"""Frozen types shared by more than one layer.

`api/schemas.py` is the contract between the API and the UI, but some of its models are
also produced outside the API: `outreach/messages.py` renders `Message` and `eval/report.py`
assembles `ReportResponse`. Those live here, so neither has to import the web layer to build
its own return type. `api/schemas.py` re-exports every one of them under the same name, so to
the UI and to every existing import nothing moved. The freeze rules there apply here too.
"""

from __future__ import annotations

from leeward.contracts.base import Base
from leeward.contracts.outreach import Message
from leeward.contracts.report import (
    AblationRow,
    CalibrationBin,
    DecisionQualityRow,
    DiscriminationRow,
    FairnessRow,
    RecoveryRow,
    ReportResponse,
)

__all__ = [
    "Base", "Message",
    "ReportResponse", "RecoveryRow", "CalibrationBin", "FairnessRow",
    "AblationRow", "DecisionQualityRow", "DiscriminationRow",
]

"""τ[a,k] and the lead time of every action (SPEC §7.2, docs/proposal.md §6).

    load(path)  -> {action: {need: tau}}      how much of need k the action prevents
    leads(path) -> {action: lead_days}        how far ahead of the risk day it must happen

Both read `tau.yaml`, which keeps one row per action so a clinician retunes the prevention
fraction and the lead time in the same place. They are returned separately because they are
different kinds of number and every caller wants exactly one of them: `load()` feeds the EHA
arithmetic and is validated as a probability, `leads()` feeds the schedule and is validated
as a whole number of days.

`check_in_call` is not in the table. It prevents nothing by itself; it finds out, and its
value is the information it buys (`eha.voi`). Its lead is `INFORMATION_LEAD_DAYS`, zero and
not editable, because a find-out call that is not made today cannot re-score today.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from leeward.schema import ACTIONS, NEEDS

PATH = Path(__file__).with_name("tau.yaml")

#: Actions whose value is information, not prevention. They never appear in tau.yaml.
INFORMATION_ACTIONS = ("check_in_call",)

#: An information action is worth having only on the day it can change the answer.
INFORMATION_LEAD_DAYS = 0

#: The lead-time key inside each row. Everything else in a row must be a need.
LEAD_KEY = "lead_days"


def _rows(path: Path | str | None) -> dict[str, tuple[dict[str, float], int]]:
    """Every row of tau.yaml, validated once: (tau per need, lead days) per action."""
    path = Path(path) if path is not None else PATH
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected a mapping of action -> {{need: tau, lead_days: n}}")
    out: dict[str, tuple[dict[str, float], int]] = {}
    for action, row in sorted(raw.items()):
        if action not in ACTIONS:
            raise ValueError(f"{path}: unknown action {action!r}; actions are {ACTIONS}")
        if action in INFORMATION_ACTIONS:
            raise ValueError(f"{path}: {action} is an information action; it has no tau row")
        if not isinstance(row, dict) or set(row) != {*NEEDS, LEAD_KEY}:
            raise ValueError(f"{path}: {action} must give tau for exactly {NEEDS}, "
                             f"and a {LEAD_KEY}")
        for k in NEEDS:
            v = row[k]
            if isinstance(v, bool) or not isinstance(v, int | float) or not 0 <= v <= 1:
                raise ValueError(f"{path}: tau[{action}, {k}] must be in [0, 1], got {v!r}")
        lead = row[LEAD_KEY]
        if isinstance(lead, bool) or not isinstance(lead, int) or lead < 0:
            raise ValueError(f"{path}: {LEAD_KEY} for {action} must be a whole number of "
                             f"days, zero or more, got {lead!r}")
        out[action] = ({k: float(row[k]) for k in NEEDS}, lead)
    return out


def load(path: Path | str | None = None) -> dict[str, dict[str, float]]:
    """τ per prevention action, each row keyed in canonical `NEEDS` order."""
    return {a: row for a, (row, _) in _rows(path).items()}


def leads(path: Path | str | None = None) -> dict[str, int]:
    """Lead days per action `allocate()` can propose, information actions included.

    An action for a risk on day t is done on day `t - leads()[action]`. That day, not t, is
    the one it spends a unit of the care team's capacity on.
    """
    out = {a: lead for a, (_, lead) in _rows(path).items()}
    return out | dict.fromkeys(INFORMATION_ACTIONS, INFORMATION_LEAD_DAYS)


def matrix(table: dict[str, dict[str, float]]) -> tuple[list[str], np.ndarray]:
    """(actions, A x K array) with rows in the order of `actions` and columns in `NEEDS`."""
    actions = sorted(table)
    return actions, np.array([[table[a][k] for k in NEEDS] for a in actions], dtype=float)

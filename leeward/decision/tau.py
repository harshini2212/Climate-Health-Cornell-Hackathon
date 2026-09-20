"""What each action prevents, and when it has to happen (SPEC §7.2, §7.4).

    τ[a,k]         the fraction of need k that action a prevents, if it is done
    lead_days[a]   the last day a still works, counted back from the day the risk lands

Both are clinician-editable judgements of the same kind, so both live in `tau.yaml` and are
validated as hard as the severity weights, for the same reason: a typo that silently reorders
a care-team list, or silently moves an action a day, is worse than a crash.

`check_in_call` has no τ row. It prevents nothing by itself; it finds out, and its value is
the information it buys (`eha.voi`). It does have a lead time -- zero -- because the allocator
has to schedule it like everything else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from leeward.schema import ACTIONS, NEEDS

PATH = Path(__file__).with_name("tau.yaml")

#: The two sections of tau.yaml. Nothing else may sit at the top level.
SECTIONS = ("tau", "lead_days")

#: Actions whose value is information, not prevention. They never appear in the τ table.
INFORMATION_ACTIONS = ("check_in_call",)

#: No action may need more warning than the forecast gives. The window `GET /forecast` serves
#: is seven days (SPEC §9), so a lead longer than that could never be offered.
MAX_LEAD_DAYS = 7


def _section(path: Path, name: str) -> dict:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or set(raw) != set(SECTIONS):
        raise ValueError(f"{path}: expected exactly the top-level sections {list(SECTIONS)}, "
                         f"got {sorted(raw) if isinstance(raw, dict) else type(raw).__name__}")
    body = raw[name]
    if not isinstance(body, dict) or not body:
        raise ValueError(f"{path}: section {name!r} must be a non-empty mapping")
    return body


def load(path: Path | str | None = None) -> dict[str, dict[str, float]]:
    """τ per prevention action, each row keyed in canonical `NEEDS` order."""
    path = Path(path) if path is not None else PATH
    out: dict[str, dict[str, float]] = {}
    for action, row in sorted(_section(path, "tau").items()):
        if action not in ACTIONS:
            raise ValueError(f"{path}: unknown action {action!r}; actions are {ACTIONS}")
        if action in INFORMATION_ACTIONS:
            raise ValueError(f"{path}: {action} is an information action; it has no tau row")
        if not isinstance(row, dict) or set(row) != set(NEEDS):
            raise ValueError(f"{path}: {action} must give tau for exactly {NEEDS}")
        for k in NEEDS:
            v = row[k]
            if isinstance(v, bool) or not isinstance(v, int | float) or not 0 <= v <= 1:
                raise ValueError(f"{path}: tau[{action}, {k}] must be in [0, 1], got {v!r}")
        out[action] = {k: float(row[k]) for k in NEEDS}
    return out


def lead_days(path: Path | str | None = None) -> dict[str, int]:
    """Lead time in days per action -- every action in `ACTIONS`, none missing.

    Missing is not the same as zero, so a new action added to the schema without a judgement
    about when it has to happen fails here rather than being quietly scheduled for today.
    """
    path = Path(path) if path is not None else PATH
    raw = _section(path, "lead_days")
    if set(raw) != set(ACTIONS):
        missing, unknown = sorted(set(ACTIONS) - set(raw)), sorted(set(raw) - set(ACTIONS))
        raise ValueError(f"{path}: lead_days must name exactly {len(ACTIONS)} actions; "
                         f"missing {missing}, unknown {unknown}")
    out: dict[str, int] = {}
    for action, v in raw.items():
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= MAX_LEAD_DAYS:
            raise ValueError(f"{path}: lead_days[{action}] must be a whole number of days in "
                             f"[0, {MAX_LEAD_DAYS}], got {v!r}")
        out[action] = int(v)
    return out


def matrix(table: dict[str, dict[str, float]]) -> tuple[list[str], np.ndarray]:
    """(actions, A x K array) with rows in the order of `actions` and columns in `NEEDS`."""
    actions = sorted(table)
    return actions, np.array([[table[a][k] for k in NEEDS] for a in actions], dtype=float)

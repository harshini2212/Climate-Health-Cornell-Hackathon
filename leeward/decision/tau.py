"""τ[a,k]: the fraction of need k that action a prevents, if it is done (SPEC §7.2).

Literature priors, kept in `tau.yaml` so they can be edited without touching code, and
validated as hard as the severity weights for the same reason.

`check_in_call` is not in the table. It prevents nothing by itself; it finds out, and its
value is the information it buys (`eha.voi`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from leeward.schema import ACTIONS, NEEDS

PATH = Path(__file__).with_name("tau.yaml")

#: Actions whose value is information, not prevention. They never appear in tau.yaml.
INFORMATION_ACTIONS = ("check_in_call",)


def load(path: Path | str | None = None) -> dict[str, dict[str, float]]:
    """τ per prevention action, each row keyed in canonical `NEEDS` order."""
    path = Path(path) if path is not None else PATH
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected a mapping of action -> {{need: tau}}")
    out: dict[str, dict[str, float]] = {}
    for action, row in sorted(raw.items()):
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


def matrix(table: dict[str, dict[str, float]]) -> tuple[list[str], np.ndarray]:
    """(actions, A x K array) with rows in the order of `actions` and columns in `NEEDS`."""
    actions = sorted(table)
    return actions, np.array([[table[a][k] for k in NEEDS] for a in actions], dtype=float)

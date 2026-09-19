"""w_k: how much harm need k does when it happens (SPEC §7.1).

The weights live in `severity.yaml`, next to this file, so a clinician can change them
without touching code. Nobody reads the diffs, so `load()` validates hard: a typo in the
YAML fails here rather than quietly reordering the care-team list.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from leeward.schema import NEEDS

PATH = Path(__file__).with_name("severity.yaml")


def load(path: Path | str | None = None) -> dict[str, float]:
    """Severity weight per need, keyed in canonical `NEEDS` order."""
    path = Path(path) if path is not None else PATH
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping of need -> weight")
    if set(raw) != set(NEEDS):
        raise ValueError(f"{path}: needs must be exactly {NEEDS}; missing "
                         f"{sorted(set(NEEDS) - set(raw))}, unknown {sorted(set(raw) - set(NEEDS))}")
    out = {}
    for k in NEEDS:
        v = raw[k]
        if isinstance(v, bool) or not isinstance(v, int | float) or not v > 0:
            raise ValueError(f"{path}: weight for {k!r} must be a positive number, got {v!r}")
        out[k] = float(v)
    return out


def vector(weights: dict[str, float]) -> np.ndarray:
    """The weights as an array in `NEEDS` order, for the matrix arithmetic in eha.py."""
    return np.array([weights[k] for k in NEEDS], dtype=float)

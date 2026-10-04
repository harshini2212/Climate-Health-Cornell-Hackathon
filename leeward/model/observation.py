"""What a VA record does and does not reliably carry: the observation contract scoring reads.

`docs/SPEC.md` §3.1 ends the cohort contract with `*_observed | | | nullable copies after
missingness`. Which fields have such a copy, and what the copy is called, is a fact about the
record the scorer is handed, so it lives here, beside the scorer. `leeward/cohort/missingness.py`
is the synthetic step that *produces* those gaps; it imports these names rather than the other
way round, so production code never has to import the synthetic cohort package to read them.
"""

from __future__ import annotations

import zlib

import numpy as np

#: The augmented fields a VA record does not reliably carry. Each is hidden independently.
HIDDEN_FIELDS = ("home_ac", "floor", "deployment_era")


def observed_name(field: str) -> str:
    return f"{field}_observed"


def stream(seed: int, name: str) -> np.random.Generator:
    """An independent random stream per field, keyed by name.

    Same idiom as `cohort/build.py` and `cohort/medications.py`: hiding a fourth field later
    draws from a new stream, so it cannot reshuffle which veterans are missing their floor.
    """
    return np.random.default_rng([seed, zlib.crc32(name.encode())])

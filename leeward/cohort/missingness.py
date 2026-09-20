"""Hide the fields a VA record does not reliably carry.

`docs/SPEC.md` §3.1 ends the cohort contract with `*_observed | | | nullable copies after
missingness`. This is that step. A seeded `HIDDEN_FRACTION` of each field in `HIDDEN_FIELDS`
is hidden, and every one of them gets a `<col>_observed` sibling that is null where the value
is hidden and equal to the value where it is not.

**The truth stays in the cohort.** `leeward/cohort/simulate.py` draws outcomes from it — the
world knows whether a veteran has air conditioning even when the VA does not. Scoring is what
may not look: `leeward/model/score_prior.py` reads the `_observed` copy and marginalises over
what is missing, so a veteran whose floor was never recorded gets a genuinely wider interval
in a flooding ZIP than one whose floor is known.

That is the claim the find-out tier rests on: uncertainty about *this person*, not about the
world. With every record complete, every veteran's epistemic share comes from the same prior
spread, the share is flat, and nobody is ever uncertain enough to earn a check-in call.

Each field is hidden on its own independent stream, so the gaps are not the same veterans
three times over: about half the cohort has at least one gap, a few have all three, and
roughly `HIDDEN_FRACTION` of each column is missing. That is the shape of a real registry.

One caveat worth saying out loud: of the three fields, only `home_ac` and `floor` are read by
`leeward/model/design.py` (`theta_heat_x_no_ac`, `theta_flood_x_lowfloor`). `deployment_era`
reaches the model only through `pact_presumptive`, which is recorded separately and is not
hidden here, so hiding it widens no interval today. It is hidden anyway because the contract
asks for it and because a term that reads it may land later; the scorer discovers which
fields actually move a score rather than being told.
"""

from __future__ import annotations

import zlib

import numpy as np
import polars as pl

#: Share of each field hidden, per `docs/SPEC.md` §5.5.
HIDDEN_FRACTION = 0.20

#: The augmented fields a VA record does not reliably carry. Each is hidden independently.
HIDDEN_FIELDS = ("home_ac", "floor", "deployment_era")


def stream(seed: int, name: str) -> np.random.Generator:
    """An independent random stream per field, keyed by name.

    Same idiom as `build.py` and `medications.py`: hiding a fourth field later draws from a
    new stream, so it cannot reshuffle which veterans are missing their floor.
    """
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def observed_name(field: str) -> str:
    return f"{field}_observed"


def apply(people: pl.DataFrame, seed: int = 0, *,
          fraction: float = HIDDEN_FRACTION) -> pl.DataFrame:
    """Add a nullable `<field>_observed` sibling for every field in `HIDDEN_FIELDS`."""
    out = []
    for field in HIDDEN_FIELDS:
        hidden = stream(seed, f"missing_{field}").random(people.height) < fraction
        out.append(
            pl.when(pl.Series(hidden))
              .then(pl.lit(None, dtype=people.schema[field]))
              .otherwise(pl.col(field))
              .alias(observed_name(field))
        )
    return people.with_columns(out)

"""The contract frames the tests read. Built here, never loaded from `data/`.

`data/*.parquet` is pipeline state, not test input. `make fixtures` puts 500 fake veterans
there; `make cohort` replaces some of it with 10,000 real ones; in between, the directory
holds a mix of the two. A suite that reads it is therefore testing whatever make target ran
last -- green on one machine, red on the next, for reasons that have nothing to do with the
code under review. Anyone who ran the real pipeline and then ran the gate chased that ghost.

So the suite builds its own frames. Same generators as `scripts/make_fixtures.py`, so the
shapes stay in lockstep with what the lanes develop against, but they live in memory, are
seeded once here, and no make target can reach them. `make fixtures && make check` and
`make cohort && make score && make check` now run the same tests over the same numbers.

The geography is still real: ZIPs, boroughs and facilities come from `data/reference/`,
which is committed input rather than pipeline output, so a wrong join still shows up here.
"""

from __future__ import annotations

import importlib.util
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from leeward import schema

ROOT = Path(__file__).resolve().parents[1]

#: Deliberately the defaults of `scripts/make_fixtures.py`: the panel every lane was built
#: against. Changing these changes the numbers several hand-checked tests expect.
VETERANS = 500
DAYS = 30
SEED = 0
START = date(2026, 7, 1)

_CACHE: dict[str, pl.DataFrame] = {}


def _generators():
    """`scripts/make_fixtures.py`, imported by path -- `scripts/` is not a package.

    Reusing it rather than copying it is the point: one definition of what a correctly
    shaped table looks like, so the tests cannot quietly drift from what the lanes get
    from `make fixtures`. The cost is that renaming a `make_*` function there breaks this.
    """
    spec = importlib.util.spec_from_file_location(
        "leeward_fixture_generators", ROOT / "scripts" / "make_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build() -> dict[str, pl.DataFrame]:
    gen = _generators()
    _, _, zips, fac = gen._geography()
    days = [START + timedelta(days=i) for i in range(DAYS)]

    cohort = gen.make_cohort(VETERANS, SEED, zips, fac)
    scores = gen.make_scores(cohort, days, SEED)
    actions = gen.make_actions(cohort, scores, days[DAYS // 2], SEED)
    built = {
        "cohort": cohort,
        "hazards": gen.make_hazards(zips, days, SEED),
        "site_status": gen.make_site_status(fac, days),
        "scores": scores,
        "outcomes": gen.make_outcomes(scores, SEED),
        "actions": actions,
        "outcome_log": gen.make_outcome_log(actions, SEED),
    }
    missing = set(schema.TABLES) - set(built)
    assert not missing, f"no test frame for contract table(s): {sorted(missing)}"
    # `schema.write()` is the rule for contract tables because it validates first -- but it
    # also writes to `data/`, which is exactly what this module exists to avoid. So validate
    # here instead: a shape drift still fails loudly, and nothing touches the disk.
    return {name: schema.validate(df, name) for name, df in built.items()}


def table(name: str) -> pl.DataFrame:
    """One contract table, built once per session. Never reads `data/<name>.parquet`."""
    if not _CACHE:
        _CACHE.update(_build())
    if name not in _CACHE:
        raise KeyError(f"{name!r} is not a contract table; have {sorted(_CACHE)}")
    return _CACHE[name]

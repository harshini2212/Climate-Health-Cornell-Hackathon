"""Rung 0: score every veteran-day-need straight from the priors. No MCMC.

This is the demo's floor, and it stays reachable by name at every rung above it:

    python -m leeward.model.score_prior                          # every date in hazards
    python -m leeward.model.score_prior --start 2026-08-29 --days 7

The machinery moved to `score.py` when rung 1 landed, because the two differ in exactly one
thing -- where the coefficient draws come from -- and two copies of a matrix multiply are
two chances to score the demo from one set of numbers and the report from another. This
module is now the prior-only entry point: `score.py --prior`, spelled the way the rung-0
task and `docs/PROMPTS.md` spell it.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import date

import polars as pl

from leeward.model import priors
from leeward.model import score as _score

RUNG = 0
N_DRAWS = _score.N_DRAWS
SEED = _score.SEED


def score(cohort: pl.DataFrame, hazards: pl.DataFrame, site_status: pl.DataFrame, *,
          dates: Iterable[date] | None = None, n_draws: int = N_DRAWS, seed: int = SEED,
          scale: float = 1.0) -> pl.DataFrame:
    """Rung-0 scores for every veteran in `cohort` on every date in `dates`.

    `hazards` may reach back before `dates`; those days feed the lag curves, and its first
    day is when `days_supply_remaining` was counted. `scale` is the prior-scale multiplier.
    """
    coef = _score.Coefficients(
        B=priors.draw(n_draws, seed=seed, scale=scale), rung=RUNG, source="prior",
        detail=f"rung 0 priors, {n_draws} draws, seed {seed}, scale {scale}")
    return _score.score(cohort, hazards, site_status, dates=dates, coef=coef)


def _passthrough(args: argparse.Namespace) -> list[str]:
    out = []
    if args.start:
        out += ["--start", args.start.isoformat()]
    if args.days:
        out += ["--days", str(args.days)]
    return out + ["--draws", str(args.draws), "--seed", str(args.seed),
                  "--scale", str(args.scale), "--prior"]


def main(argv: list[str] | None = None) -> int:
    ap = _score.add_arguments(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    return _score.main(_passthrough(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())

"""`make score` -- coefficients x today's hazards -> `data/scores.parquet`. Any rung.

    python -m leeward.model.score                          # every date in hazards
    python -m leeward.model.score --start 2026-08-29 --days 7
    python -m leeward.model.score --prior                  # ignore the posterior, rung 0

The coefficients come from `data/posterior.nc` when a fit exists and from `priors.py` when
one does not, and **the rung on every row says which**. That is the whole reason the column
is in the contract: a clean clone has no posterior, so `make demo` scores from the priors at
rung 0 and says rung 0 on stage; `make fit` then raises every number to rung 1 with nothing
downstream changed. Nothing here runs inference -- the posterior is read, thinned and
multiplied (SPEC 1: never run inference inside a request).

Per veteran, date and need, over the draws of p = sigmoid(X @ B):

    p_mean              mean
    p_lo80, p_hi80      10th and 90th percentiles
    p_epistemic_share   Var(p) / (Var(p) + E[p(1-p)])  -- equal to Var(p) / (p_mean(1-p_mean))
    driver_1..3         the three terms with the largest |contribution| to the mean
                        log-odds, from design.term_contributions; the intercept is never one

When the draws are very skewed, the mean of p can fall outside its own 10-90 band (roughly
once the log-odds spread exceeds 2.6). The band is then widened to include the mean, so
p_lo80 <= p_mean <= p_hi80 always holds; `main` prints how many rows that touched.

At rung 1 the 16 interaction coefficients are still prior draws -- the fit did not see them
(`hazard.py`) -- so a veteran card's medication and SiteDown drivers are as wide as they
were at rung 0. `report/fit.json` names every term in both sets.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from leeward import schema
from leeward.model import design, hazard, priors
from leeward.schema import NEEDS

N_DRAWS = 400               # SPEC 6.4: thin the posterior to 400 draws
SEED = 0
LO_Q, HI_Q = 0.10, 0.90
DAYS_PER_BUILD = 7          # design matrix built a week at a time
BLOCK_ROWS = 512            # 512 rows x 5 needs x 400 draws = 8 MB per thread
ETA_CLIP = 30.0             # keeps sigmoid strictly inside (0, 1) in float64
MIN_DRIVER_LOGODDS = 0.05   # a term below this is not worth a chip on the card
_THREADS = min(8, os.cpu_count() or 1)

_K = design.K
_INTERCEPT = [t.name for t in design.TERMS].index("alpha")
_DRIVER_TERMS = [j for j in range(len(design.TERMS)) if j != _INTERCEPT]
_PHRASES = [design.TERMS[j].phrase for j in _DRIVER_TERMS]


# --------------------------------------------------------------------------- #
# Where the coefficients come from
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class Coefficients:
    """(D, P, K) draws, the rung they came from, and one word saying which."""
    B: np.ndarray
    rung: int
    source: str             # "posterior" or "prior"
    detail: str             # one line for the CLI, so a run says what it scored from


def _thin(B: np.ndarray, n_draws: int) -> np.ndarray:
    """Evenly spaced draws, first and last included. Deterministic: no resampling, so the
    same posterior always thins to the same 400 vectors."""
    if n_draws >= B.shape[0]:
        return B
    return B[np.linspace(0, B.shape[0] - 1, n_draws).round().astype(int)]


def coefficients(*, posterior: Path | None = hazard.POSTERIOR, n_draws: int = N_DRAWS,
                 seed: int = SEED, scale: float = 1.0) -> Coefficients:
    """The posterior if `make fit` has run, the priors otherwise. Never silently mixed.

    **A term the fit did not see is not in the prediction.** The posterior file carries the
    carried terms at their priors, because that is what a coefficient no likelihood touched
    is worth and `eval/recovery.py` has to be able to read every term. But the fitted main
    effects are *marginal over* the terms left out: with no interaction columns in the
    likelihood, rung 1's `alpha (heat)` comes back near -4.4 against a generating -6.0
    precisely because it has absorbed the six heat interactions. Adding the prior draws for
    those interactions back on top of it would count them twice and score every hot day
    roughly four times too high. So scoring uses exactly the terms the rung fitted, and
    `report/fit.json` names the ones it did not.
    """
    if posterior is not None and Path(posterior).exists():
        post = hazard.read_posterior(posterior)
        if scale != 1.0:
            raise ValueError(
                f"prior_scale {scale} cannot be applied to a fitted posterior; the slider's "
                "0.5 and 2.0 are prior-only (rung 0). Pass --prior, or refit with --scale.")
        B = _thin(post.B, n_draws).copy()
        for name in post.carried:
            B[:, design.TERM_SLICE[name], :] = 0.0
        carried = f"; {len(post.carried)} interaction terms not in this rung" if post.carried else ""
        return Coefficients(B=B, rung=post.rung, source="posterior",
                            detail=f"rung {post.rung} posterior, {B.shape[0]} of "
                                   f"{post.n_draws} draws{carried}")
    B = priors.draw(n_draws, seed=seed, scale=scale)
    return Coefficients(B=B, rung=0, source="prior",
                        detail=f"rung 0 priors, {n_draws} draws, seed {seed}, scale {scale}")


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _summarise(d: design.Design, B_flat: np.ndarray, B_mean: np.ndarray,
               n_draws: int) -> pl.DataFrame:
    """Scores for one design matrix, one row per (row, need). Strings are attached later:
    carrying veteran ids and phrases as integers keeps a 10k x 120-day run small."""
    n = d.X.shape[0]
    mean, lo, hi, share = (np.empty((n, _K)) for _ in range(4))

    def block(s: int) -> None:
        # Each block writes only its own rows, so thread scheduling cannot change a number.
        e = min(s + BLOCK_ROWS, n)
        eta = (d.X[s:e] @ B_flat).reshape(e - s, _K, n_draws)
        np.clip(eta, -ETA_CLIP, ETA_CLIP, out=eta)
        p = 1.0 / (1.0 + np.exp(-eta))
        m = p.mean(axis=-1)
        q = np.quantile(p, [LO_Q, HI_Q], axis=-1)
        mean[s:e] = m
        lo[s:e] = np.minimum(q[0], m)
        hi[s:e] = np.maximum(q[1], m)
        share[s:e] = np.clip(p.var(axis=-1) / (m * (1.0 - m)), 0.0, 1.0)

    # numpy drops the GIL for the exp and the percentile partition, which are most of the cost.
    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        list(pool.map(block, range(0, n, BLOCK_ROWS)))

    contrib = design.term_contributions(d.X, B_mean)[:, _DRIVER_TERMS, :]   # (N, T', K)
    contrib = contrib.transpose(0, 2, 1).reshape(n * _K, -1)                 # (N*K, T')
    top = np.argsort(-np.abs(contrib), axis=1, kind="stable")[:, :3]
    val = np.take_along_axis(contrib, top, axis=1)
    keep = np.abs(val) >= MIN_DRIVER_LOGODDS
    code = np.where(keep, top, -1)      # -1: no driver, a null rather than a made-up reason

    out = {
        "_i": pl.Series(np.repeat(d.vet_index, _K), dtype=pl.UInt32),
        "date": pl.Series(np.repeat(d.date, _K), dtype=pl.Date),
        "_k": pl.Series(np.tile(np.arange(_K), n), dtype=pl.Int8),
        "p_mean": mean.ravel(),
        "p_lo80": lo.ravel(),
        "p_hi80": hi.ravel(),
        "p_epistemic_share": share.ravel(),
    }
    for i in range(3):
        out[f"_d{i + 1}"] = pl.Series(code[:, i], dtype=pl.Int32)
        out[f"driver_{i + 1}_contrib"] = pl.Series(np.where(keep[:, i], val[:, i], np.nan),
                                                   nan_to_null=True)
    return pl.DataFrame(out)


def _decode(col: str, labels: list[str]) -> pl.Expr:
    return pl.col(col).replace_strict(list(range(len(labels))), labels, default=None,
                                      return_dtype=pl.Utf8)


def score(cohort: pl.DataFrame, hazards: pl.DataFrame, site_status: pl.DataFrame, *,
          dates: Iterable[date] | None = None, coef: Coefficients | None = None,
          n_draws: int = N_DRAWS, seed: int = SEED, scale: float = 1.0,
          posterior: Path | None = hazard.POSTERIOR) -> pl.DataFrame:
    """Scores for every veteran in `cohort` on every date in `dates`, at whatever rung the
    coefficients came from.

    `hazards` may reach back before `dates`; those days feed the lag curves, and its first
    day is when `days_supply_remaining` was counted. Pass `coef` to score a panel twice
    without reading the posterior twice.
    """
    dates = sorted(hazards["date"].unique().to_list()) if dates is None else sorted(dates)
    ref = hazards["date"].min()
    if coef is None:
        coef = coefficients(posterior=posterior, n_draws=n_draws, seed=seed, scale=scale)

    n_draws = coef.B.shape[0]
    B_flat = np.ascontiguousarray(coef.B.transpose(1, 2, 0).reshape(design.P, _K * n_draws))
    B_mean = coef.B.mean(axis=0)

    parts = []
    for i in range(0, len(dates), DAYS_PER_BUILD):
        d = design.build(cohort, hazards, site_status,
                         dates=dates[i:i + DAYS_PER_BUILD], ref_date=ref)
        parts.append(_summarise(d, B_flat, B_mean, n_draws))

    if not parts:
        return schema.empty("scores")
    df = pl.concat(parts)
    return df.select(
        cohort["veteran_id"].gather(df["_i"]),
        "date",
        _decode("_k", NEEDS).alias("need"),
        "p_mean", "p_lo80", "p_hi80", "p_epistemic_share",
        *[_decode(f"_d{i}", _PHRASES).alias(f"driver_{i}") for i in (1, 2, 3)],
        *[f"driver_{i}_contrib" for i in (1, 2, 3)],
        model_rung=pl.lit(coef.rung, dtype=pl.Int32),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--start", type=date.fromisoformat, help="first date to score")
    ap.add_argument("--days", type=int, help="how many dates to score from --start")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--scale", type=float, default=1.0, help="prior-scale multiplier (rung 0)")
    ap.add_argument("--prior", action="store_true",
                    help="score from the priors even if data/posterior.nc exists (rung 0)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = add_arguments(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    args = ap.parse_args(argv)

    cohort, hazards = schema.read("cohort"), schema.read("hazards")
    sites = schema.read("site_status")
    dates = sorted(hazards["date"].unique().to_list())
    if args.start:
        dates = [d for d in dates if d >= args.start]
    if args.days:
        dates = dates[:args.days]
    if not dates:
        print("no hazard dates in the requested window", file=sys.stderr)
        return 1

    coef = coefficients(posterior=None if args.prior else hazard.POSTERIOR,
                        n_draws=args.draws, seed=args.seed, scale=args.scale)
    t0 = time.perf_counter()
    df = score(cohort, hazards, sites, dates=dates, coef=coef)
    path = schema.write(df, "scores")
    secs = time.perf_counter() - t0

    widened = df.filter((pl.col("p_lo80") == pl.col("p_mean")) |
                        (pl.col("p_hi80") == pl.col("p_mean"))).height
    print(f"{coef.detail}: {cohort.height:,} veterans x {len(dates)} days x {len(NEEDS)} "
          f"needs = {df.height:,} rows, {secs:.1f}s -> {path}")
    print(f"  dates {dates[0]} .. {dates[-1]}; model_rung {coef.rung}; 10-90 band widened to "
          f"reach the mean on {widened:,} rows")
    summary = (df.group_by("need")
                 .agg(pl.col("p_mean").mean().alias("mean"),
                      pl.col("p_mean").quantile(0.99).alias("p99"),
                      pl.col("p_mean").max().alias("max"),
                      (pl.col("p_epistemic_share") >= 0.4).mean().alias("share>=0.4"))
                 .sort(pl.col("need").replace_strict(NEEDS, list(range(len(NEEDS))))))
    for r in summary.iter_rows(named=True):
        print(f"  {r['need']:14s} mean p {r['mean']:.4f}  p99 {r['p99']:.4f}  "
              f"max {r['max']:.4f}  epistemic share >= 0.4 on {r['share>=0.4']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

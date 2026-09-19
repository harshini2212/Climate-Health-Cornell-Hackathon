"""Rung 0: score every veteran-day-need straight from the priors. No MCMC.

This is the demo's floor. 400 coefficient vectors drawn from `priors.py`, one matrix
multiply per block of rows through `design.py`, and `data/scores.parquet` written through
`schema.write` with `model_rung = 0`. Every rung above this writes the same table.

    python -m leeward.model.score_prior                          # every date in hazards
    python -m leeward.model.score_prior --start 2026-08-29 --days 7

Per veteran, date and need, over the draws of p = sigmoid(X @ B):

    p_mean              mean
    p_lo80, p_hi80      10th and 90th percentiles
    p_epistemic_share   Var(p) / (Var(p) + E[p(1-p)])  -- equal to Var(p) / (p_mean(1-p_mean))
    driver_1..3         the three terms with the largest |contribution| to the mean
                        log-odds, from design.term_contributions; the intercept is never one

When the draws are very skewed, the mean of p can fall outside its own 10-90 band (roughly
once the log-odds spread exceeds 2.6). The band is then widened to include the mean, so
p_lo80 <= p_mean <= p_hi80 always holds; `main` prints how many rows that touched.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import numpy as np
import polars as pl

from leeward import schema
from leeward.model import design, priors
from leeward.schema import NEEDS

RUNG = 0
N_DRAWS = 400
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
          dates: Iterable[date] | None = None, n_draws: int = N_DRAWS, seed: int = SEED,
          scale: float = 1.0) -> pl.DataFrame:
    """Rung-0 scores for every veteran in `cohort` on every date in `dates`.

    `hazards` may reach back before `dates`; those days feed the lag curves, and its first
    day is when `days_supply_remaining` was counted. `scale` is the prior-scale multiplier.
    """
    dates = sorted(hazards["date"].unique().to_list()) if dates is None else sorted(dates)
    ref = hazards["date"].min()

    B = priors.draw(n_draws, seed=seed, scale=scale)                        # (D, P, K)
    B_flat = np.ascontiguousarray(B.transpose(1, 2, 0).reshape(design.P, _K * n_draws))
    B_mean = B.mean(axis=0)

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
        model_rung=pl.lit(RUNG, dtype=pl.Int32),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=date.fromisoformat, help="first date to score")
    ap.add_argument("--days", type=int, help="how many dates to score from --start")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
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

    t0 = time.perf_counter()
    df = score(cohort, hazards, sites, dates=dates, n_draws=args.draws, seed=args.seed)
    path = schema.write(df, "scores")
    secs = time.perf_counter() - t0

    widened = df.filter((pl.col("p_lo80") == pl.col("p_mean")) |
                        (pl.col("p_hi80") == pl.col("p_mean"))).height
    print(f"rung {RUNG} (prior-only, no MCMC): {cohort.height:,} veterans x {len(dates)} days "
          f"x {len(NEEDS)} needs = {df.height:,} rows, {args.draws} draws, seed {args.seed}, "
          f"{secs:.1f}s -> {path}")
    print(f"  dates {dates[0]} .. {dates[-1]}; 10-90 band widened to reach the mean on "
          f"{widened:,} rows")
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

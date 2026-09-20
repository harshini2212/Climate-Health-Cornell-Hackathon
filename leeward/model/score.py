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
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from leeward import schema
from leeward.cohort import missingness
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

@dataclass(frozen=True)
class _Unknown:
    """One hidden field: who is missing it, what it could be, and how likely each is."""
    field: str
    hidden: np.ndarray      # (n_veterans,) bool
    values: tuple           # candidate values; values[0] is the stand-in in the seen cohort
    #: (n_veterans, n_values) -- P(value | this veteran's observed covariates), estimated
    #: from the veterans who do have the field. Per veteran, not one population rate: see
    #: `_conditional_weights`.
    weights: np.ndarray
    #: the columns that rate was conditioned on, for `main` to print
    given: tuple = ()


@dataclass(frozen=True)
class _Margin:
    """One hidden field, resolved against one week's design matrix."""
    unknown: _Unknown
    pick: np.ndarray        # (n_veterans, n_draws) which candidate each draw assumes
    #: per candidate after the stand-in: (columns that move, their delta) or None
    deltas: list


def _design_reads(field: str) -> bool:
    """Does any design term actually read this column?

    Asked of `design.TERMS` rather than hardcoded, so a term that starts reading a hidden
    field tomorrow is marginalised over without an edit here -- and a field the design does
    not read (today, `deployment_era`, which reaches the model only through the separately
    recorded `pact_presumptive`) costs nothing.
    """
    return any(field in e.meta.root_names() for t in design.TERMS for e in t.exprs)


def _unknowns(cohort: pl.DataFrame) -> list[_Unknown]:
    """The hidden fields worth marginalising over, with their population rates.

    Rates come from the veterans whose value *is* on file, not from a constant in this file:
    the scorer is not allowed to know the truth column, and a hand-written rate would drift
    from the cohort the moment either changed. They are conditioned on what the record does
    have, per veteran, rather than pooled into one population number -- `_conditional_weights`
    says why that distinction is not cosmetic here.
    """
    out = []
    for field in missingness.HIDDEN_FIELDS:
        obs = missingness.observed_name(field)
        if obs not in cohort.columns or not _design_reads(field):
            continue
        hidden = cohort[obs].is_null().to_numpy()
        seen = cohort[obs].drop_nulls()
        if not hidden.any() or seen.is_empty():
            continue
        values = tuple(seen.value_counts(sort=True)[obs].to_list())
        weights, given = _conditional_weights(cohort, obs, values)
        out.append(_Unknown(field, hidden, values, weights, given))
    return out


def _seen(cohort: pl.DataFrame) -> pl.DataFrame:
    """The cohort as scoring is allowed to see it: every hidden field replaced by its
    `_observed` copy, with the most common observed value standing in where it is missing.

    The truth columns stay in the frame -- `cohort/simulate.py` needs them -- but after this
    the design cannot reach them, because the name it reads now holds the observed copy.
    """
    exprs = []
    for field in missingness.HIDDEN_FIELDS:
        obs = missingness.observed_name(field)
        if obs not in cohort.columns:
            continue
        seen = cohort[obs].drop_nulls()
        if seen.is_empty():
            continue
        stand_in = seen.value_counts(sort=True)[obs][0]
        exprs.append(pl.col(obs).fill_null(stand_in).alias(field))
    return cohort.with_columns(exprs) if exprs else cohort


def _draw_unknowns(unknowns: list[_Unknown], n_draws: int, seed: int) -> list[np.ndarray]:
    """Per veteran and per draw, which candidate value this draw assumes. 0 = the stand-in,
    so a veteran whose value is on file contributes no delta at all."""
    picks = []
    for u in unknowns:
        rng = missingness.stream(seed, f"marginalise_{u.field}")
        pick = np.zeros((u.hidden.size, n_draws), dtype=np.int8)
        # Each veteran draws from their own conditional distribution, so this is an inverse
        # CDF per row rather than one `rng.choice(p=...)` over a shared vector.
        cdf = np.cumsum(u.weights[u.hidden], axis=1)            # (n_hidden, n_values)
        cdf[:, -1] = 1.0
        draw = rng.random((int(u.hidden.sum()), n_draws))
        pick[u.hidden] = (draw[:, :, None] >= cdf[:, None, :]).sum(axis=2).astype(np.int8)
        picks.append(pick)
    return picks


def _deltas(seen: pl.DataFrame, u: _Unknown, X0: np.ndarray,
            build: Callable[[pl.DataFrame], np.ndarray],
            ) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """For each candidate value after the stand-in, the change it makes to the design matrix.

    The design is rebuilt with the hidden veterans' field forced to that value and differenced
    against the stand-in build `X0`, so this stays correct whatever expression the term uses.
    Only the columns that actually move are kept -- for `home_ac` that is one column out of P.
    """
    out: list[tuple[np.ndarray, np.ndarray] | None] = []
    for value in u.values[1:]:
        alt = seen.with_columns(
            pl.when(pl.Series(u.hidden))
              .then(pl.lit(value, dtype=seen.schema[u.field]))
              .otherwise(pl.col(u.field)).alias(u.field))
        delta = build(alt) - X0
        cols = np.flatnonzero(np.abs(delta).max(axis=0) > 0)
        out.append((cols, np.ascontiguousarray(delta[:, cols])) if cols.size else None)
    return out


def _conditional_weights(cohort: pl.DataFrame, obs: str,
                         values: tuple) -> tuple[np.ndarray, tuple[str, ...]]:
    """P(value | this veteran's observed covariates), estimated off the records that have it.

    Imputing from the *unconditional* rate is the textbook mistake, and here it is not a
    small one. `home_ac` is missing for a fifth of the panel, and among the records that
    have it, P(no AC) runs from 0.04 in a cool, resourced ZIP to 0.31 in a hot one where
    PLACES says people cannot afford to run what they own -- an eightfold spread that a
    single population rate flattens. Flattening it would understate the risk of exactly the
    veterans the fairness audit watches, and it would do it by throwing away per-ZIP numbers
    that are sitting in `data/reference/`, which `CLAUDE.md` forbids outright.

    Worth knowing before reading a baseline diff against this: `harm_averted_k40` is not
    precise to a percent. Marginal, one conditioning column and two give 16.31, 16.25 and
    16.14 -- all three defensible, spanning 1% -- off `p_mean` changes averaging 5e-05. The
    allocator is greedy under a capacity cap, so a hair's-breadth reordering reshuffles who
    makes the cut. `CONDITION_COLS` is set from which one recovers the true per-cell rate,
    not from which one scores best; picking it off the metric would be fitting the metric.

    So: pick the `CONDITION_COLS` observed columns whose cells move the rate most, and read
    the rate off those cells, shrunk toward the marginal by `CONDITION_SHRINK` pseudo-counts
    so a thin cell cannot invent a confident number. The veterans being imputed have these
    covariates -- that is what makes them conditionable -- so every row gets a cell.
    """
    seen = cohort.filter(pl.col(obs).is_not_null())
    # Values are carried as their index in `values`, never as text: polars renders a boolean
    # as "true" where Python renders it "True", and a mismatch there fails silently by
    # missing every cell and falling back to the marginal -- which looks exactly like a
    # field that genuinely does not vary. It did.
    code = {v: i for i, v in enumerate(values)}
    seen_code = seen[obs].replace_strict(list(code), list(code.values()),
                                         return_dtype=pl.Int32)
    counts = np.bincount(seen_code.to_numpy(), minlength=len(values)).astype(float)
    marginal = counts / counts.sum()

    usable = [c for c in CONDITION_ON if c in cohort.columns and cohort[c].n_unique() > 1]
    keys = {c: _bin(cohort[c]) for c in usable}
    is_seen = cohort[obs].is_not_null()

    def spread(col: str) -> float:
        """How much this column moves the rate: the cell rates' count-weighted variance."""
        cell = (pl.DataFrame({"k": keys[col].filter(is_seen), "v": seen_code})
                  .group_by("k").agg((pl.col("v") == 0).mean().alias("p"),
                                     pl.len().alias("n")))
        p, n = cell["p"].to_numpy(), cell["n"].to_numpy().astype(float)
        return float(np.average((p - np.average(p, weights=n)) ** 2, weights=n))

    given = tuple(sorted(usable, key=spread, reverse=True)[:CONDITION_COLS])
    if not given:
        return np.tile(marginal, (cohort.height, 1)), ()

    key_all = (pl.DataFrame({c: keys[c] for c in given})
                 .select(pl.concat_str(given, separator="|").alias("k"))["k"])
    tab = (pl.DataFrame({"k": key_all.filter(is_seen), "v": seen_code})
             .group_by("k", "v").len()
             .pivot(on="v", index="k", values="len")
             .fill_null(0))

    cells = {k: i for i, k in enumerate(tab["k"].to_list())}
    raw = np.stack([tab[str(i)].to_numpy().astype(float) if str(i) in tab.columns
                    else np.zeros(tab.height) for i in range(len(values))], axis=1)
    # Shrink toward the marginal, then normalise: a cell of three people stays near it.
    smoothed = raw + CONDITION_SHRINK * marginal
    smoothed /= smoothed.sum(axis=1, keepdims=True)

    idx = np.array([cells.get(k, -1) for k in key_all.to_list()])
    out = np.tile(marginal, (cohort.height, 1))
    out[idx >= 0] = smoothed[idx[idx >= 0]]
    return out, given


CONDITION_ON = ("hvi", "low_assets", "transport_barrier", "mobility_impaired",
                "stormwater_flooded_frac", "evac_zone", "income_band", "borough", "age")


CONDITION_COLS = 2          # hvi x low_assets is the real structure; 1 misses low_assets


CONDITION_BINS = 4          # quartiles for the continuous ones
CONDITION_SHRINK = 20.0     # pseudo-counts pulling a thin cell back toward the marginal


def _bin(s: pl.Series) -> pl.Series:
    """A column as a conditioning key: continuous ones cut into quartiles, the rest as-is."""
    if s.dtype.is_numeric() and s.n_unique() > CONDITION_BINS * 2:
        r = s.rank("ordinal") * CONDITION_BINS // (s.len() + 1)
        return r.cast(pl.Utf8)
    return s.cast(pl.Utf8).fill_null("?")


def _summarise(d: design.Design, B_flat: np.ndarray, B_mean: np.ndarray,
               n_draws: int, margins: Sequence[_Margin] = ()) -> pl.DataFrame:
    """Scores for one design matrix, one row per (row, need). Strings are attached later:
    carrying veteran ids and phrases as integers keeps a 10k x 120-day run small."""
    n = d.X.shape[0]
    mean, lo, hi, share = (np.empty((n, _K)) for _ in range(4))
    gap_lo, gap_hi = (np.full((n, _K), np.nan) for _ in range(2))
    # A veteran with no gap at all gets null, not a range of zero width: "we know this
    # person" and "knowing more would not help" are different answers to the care team.
    has_gap = (np.zeros(n, dtype=bool) if not margins
               else np.logical_or.reduce([m.unknown.hidden[d.vet_index] for m in margins]))
    # Each candidate needs its own (rows x needs x draws) array alongside eta, so the block
    # is halved when there is anything to marginalise over. Same numbers, less peak memory --
    # and measurably faster for it: full-width blocks run 60s against 50s on the 10k panel.
    rows_per_block = BLOCK_ROWS // 2 if margins else BLOCK_ROWS

    def block(s: int) -> None:
        # Each block writes only its own rows, so thread scheduling cannot change a number.
        e = min(s + rows_per_block, n)
        eta_ref = (d.X[s:e] @ B_flat).reshape(e - s, _K, n_draws)
        eta = eta_ref.copy() if margins else eta_ref
        # The stand-in is candidate 0, so an all-zero delta is already in both extremes.
        add_lo, add_hi = (np.zeros_like(eta) for _ in range(2)) if margins else (None, None)
        for mg in margins:
            # Which candidate this draw assumes, per row of this block. A row whose field is
            # on file always picks 0, so it never enters the sum below.
            chose = mg.pick[d.vet_index[s:e]]                   # (rows, draws)
            worst, best = np.zeros_like(eta), np.zeros_like(eta)
            for c, cand in enumerate(mg.deltas, start=1):
                if cand is None:                                # this value moves no column
                    continue
                cols, delta = cand
                g = (delta[s:e] @ B_flat[cols]).reshape(e - s, _K, n_draws)
                eta += g * (chose == c)[:, None, :]
                np.minimum(best, g, out=best)
                np.maximum(worst, g, out=worst)
            add_lo += best
            add_hi += worst

        def p_of(logodds: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-np.clip(logodds, -ETA_CLIP, ETA_CLIP)))

        p = p_of(eta)
        m = p.mean(axis=-1)
        q = np.quantile(p, [LO_Q, HI_Q], axis=-1)
        mean[s:e] = m

        if margins:
            # p as it would be reported once the gaps were filled the best and the worst way.
            # Averaged over the same draws as p_mean, so the three are directly comparable.
            # sigmoid(mean eta) would not be: at these probabilities Jensen's gap between
            # mean(sigmoid) and sigmoid(mean) is larger than the effect being measured.
            #
            # Only about one row in forty has a gap the day's hazards actually give a grip
            # on -- an unrecorded floor does nothing until the water rises. On the rest the
            # deltas are identically zero, so eta is eta_ref and both ends are p_mean, which
            # is already computed. Skipping those rows is most of the cost of this column.
            moves = np.any(add_hi != add_lo, axis=(1, 2))
            g_lo, g_hi = m.copy(), m.copy()
            if moves.any():
                g_lo[moves] = p_of(eta_ref[moves] + add_lo[moves]).mean(axis=-1)
                g_hi[moves] = p_of(eta_ref[moves] + add_hi[moves]).mean(axis=-1)
            gap_lo[s:e], gap_hi[s:e] = g_lo, g_hi
        lo[s:e] = np.minimum(q[0], m)
        hi[s:e] = np.maximum(q[1], m)
        share[s:e] = np.clip(p.var(axis=-1) / (m * (1.0 - m)), 0.0, 1.0)

    # numpy drops the GIL for the exp and the percentile partition, which are most of the cost.
    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        list(pool.map(block, range(0, n, rows_per_block)))

    gap_lo[~has_gap], gap_hi[~has_gap] = np.nan, np.nan

    # Drivers are read off the *expected* design matrix, so a veteran with a hidden floor
    # gets a chip weighted by how likely a low floor is rather than by the stand-in's value.
    X_exp = d.X
    if margins:
        X_exp = d.X.copy()
        for m in margins:
            for c, cand in enumerate(m.deltas, start=1):
                if cand is not None:
                    cols, delta = cand
                    # This veteran's own P(value), not the panel's.
                    w = m.unknown.weights[d.vet_index, c][:, None]
                    X_exp[:, cols] += w * delta

    contrib = design.term_contributions(X_exp, B_mean)[:, _DRIVER_TERMS, :]   # (N, T', K)
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
        "p_gap_lo": pl.Series(gap_lo.ravel(), nan_to_null=True),
        "p_gap_hi": pl.Series(gap_hi.ravel(), nan_to_null=True),
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

    # Only the `_observed` copies from here down; `unknowns` is what is missing from them.
    # A veteran the VA has gaps on is scored by marginalising over what those fields could
    # be, which is what makes their interval genuinely wider than a veteran we know.
    seen = _seen(cohort)
    unknowns = _unknowns(cohort)
    picks = _draw_unknowns(unknowns, n_draws, seed)

    parts = []
    for i in range(0, len(dates), DAYS_PER_BUILD):
        window = dates[i:i + DAYS_PER_BUILD]

        def _X(frame: pl.DataFrame, _w: list[date] = window) -> np.ndarray:
            return design.build(frame, hazards, site_status, dates=_w, ref_date=ref).X

        d = design.build(seen, hazards, site_status, dates=window, ref_date=ref)
        margins = [_Margin(u, pick, _deltas(seen, u, d.X, _X))
                   for u, pick in zip(unknowns, picks, strict=True)]
        parts.append(_summarise(d, B_flat, B_mean, n_draws, margins))

    if not parts:
        return schema.empty("scores")
    df = pl.concat(parts)
    return df.select(
        cohort["veteran_id"].gather(df["_i"]),
        "date",
        _decode("_k", NEEDS).alias("need"),
        "p_mean", "p_lo80", "p_hi80", "p_epistemic_share", "p_gap_lo", "p_gap_hi",
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

"""The generative truth: `data/truth.json` through `design.py` -> `data/outcomes.parquet`.

    python -m leeward.cohort.simulate                  # every day in hazards.parquet
    python -m leeward.cohort.simulate --days 30 --seed 1

This is the only module in the build that knows the right answer. Everything downstream --
recovery, calibration, the fairness audit, the decision quality numbers -- is measured
against what happens here, so a bug in it is invisible: the model would recover exactly the
wrong world and agree with itself. Two things keep it honest.

**It shares `design.py` with the model.** The simulator does not own a feature mapping. It
builds `X` with `design.build`, multiplies by `design.coef_matrix(truth)` and draws. If a
term means one thing here and another in the model, `design.py` is wrong for both at once
rather than silently different between them -- which is the whole point of not forking it.

**Only two things are simulator-side**, and they enter through `design.linear_predictor`'s
`offset`, exactly where that docstring says they belong:

    zip effect     Normal(0, sigma_zip) per (MODZCTA, need)   the neighbourhood the model
                                                              has not pooled yet (SPEC 6.2's
                                                              ICAR phi, without the graph)
    frailty        Normal(0, sigma_frailty) per veteran       the per-person tendency no
                                                              record carries

Both are drawn once, from a named seed stream, and never re-drawn per block of days.

Seeding: outcomes are drawn per calendar day, from `[seed, "outcome", day]`. So simulating
one week gives that week the same outcomes it has in the full 120-day run, and the answer
cannot depend on how the work was chunked. `--seed` moves every day at once.

`truth.json` is committed. Its coefficients are the ones docs/SPEC.md 5.4 fixes, and a
`source` line for every one of them -- they are modelling assumptions, not measurements,
and are written down where a judge can ask. The medication interactions are planted even
though today's cohort has zero in every medication column: `cohort/medications.py` fills
those columns later and the terms switch on by themselves. Plant them now or the model can
never recover them.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from leeward import schema
from leeward.model import design
from leeward.schema import NEEDS, REFERENCE

TRUTH_PATH = schema.DATA / "truth.json"
SEED = 0
DAYS_PER_BUILD = 7      # one design matrix per week of dates, as score_prior.py does
ETA_CLIP = 30.0         # keeps sigmoid strictly inside (0, 1) in float64

#: A day is "off-event" when no hazard is doing anything anywhere. Outage is never exactly
#: zero in a generated scenario (there is always a little background), so it gets a floor.
QUIET_OUTAGE_FRAC = 0.05

_K = design.K


# --------------------------------------------------------------------------- #
# truth.json
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Truth:
    """The coefficients the world is generated from, plus the two latent scales."""
    coefficients: dict
    sources: dict
    sigma_zip: float
    sigma_frailty: float
    name: str = "truth"

    def coef_matrix(self) -> np.ndarray:
        """(P, K). Raises on an unknown term, a need the term does not act on, or a lag
        curve of the wrong length -- a typo cannot quietly delete a term from the world."""
        return design.coef_matrix(self.coefficients)


def load(path: str | Path = TRUTH_PATH) -> Truth:
    doc = json.loads(Path(path).read_text())
    latent = doc.get("latent", {})
    return Truth(coefficients=doc["coefficients"], sources=doc.get("sources", {}),
                 sigma_zip=float(latent["sigma_zip"]),
                 sigma_frailty=float(latent["sigma_frailty"]),
                 name=doc.get("name", Path(path).stem))


# --------------------------------------------------------------------------- #
# Randomness
# --------------------------------------------------------------------------- #

def _stream(seed: int, name: str) -> np.random.Generator:
    """One independent stream per component, keyed by name -- as `cohort/build.py` does,
    so adding a latent term later cannot reshuffle who is frail."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def _day_stream(seed: int, day: np.datetime64) -> np.random.Generator:
    """The uniforms for one calendar day. Keyed by the day itself, so a date draws the
    same outcome whatever window it was simulated in."""
    return np.random.default_rng([seed, zlib.crc32(b"outcome"),
                                  int(day.astype("datetime64[D]").astype(np.int64))])


def latent_offset(cohort: pl.DataFrame, truth: Truth, seed: int = SEED) -> np.ndarray:
    """(n_veterans, K) of what the model cannot see: this ZIP, plus this person.

    ZIP effects are keyed to the canonical 178 MODZCTAs, so a ZIP keeps its effect whether
    the whole cohort or one borough of it is simulated. Frailty is per cohort row.
    """
    zips = sorted(pl.read_parquet(REFERENCE / "nyc_modzcta.parquet")["modzcta"].to_list())
    phi = truth.sigma_zip * _stream(seed, "zip_effect").standard_normal((len(zips), _K))
    index = {z: i for i, z in enumerate(zips)}

    unknown = sorted(set(cohort["modzcta"].to_list()) - index.keys())
    if unknown:
        raise ValueError(f"cohort lives in ZIPs that are not NYC MODZCTAs: {unknown[:5]}")

    rows = np.array([index[z] for z in cohort["modzcta"].to_list()])
    frailty = truth.sigma_frailty * _stream(seed, "frailty").standard_normal(cohort.height)
    return phi[rows] + frailty[:, None]


# --------------------------------------------------------------------------- #
# The world, one block of days at a time
# --------------------------------------------------------------------------- #

def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -ETA_CLIP, ETA_CLIP)))


def _blocks(cohort: pl.DataFrame, hazards: pl.DataFrame, site_status: pl.DataFrame,
            dates: list[date], truth: Truth, seed: int, latent: bool,
            days_per_build: int) -> Iterator[tuple[design.Design, np.ndarray]]:
    """Yield (design, p) for each block of days. p is (rows, K) daily probabilities."""
    B = truth.coef_matrix()
    off = latent_offset(cohort, truth, seed) if latent else None
    ref = hazards["date"].min()          # the day days_supply_remaining was counted on
    for i in range(0, len(dates), days_per_build):
        d = design.build(cohort, hazards, site_status,
                         dates=dates[i:i + days_per_build], ref_date=ref)
        eta = design.linear_predictor(d.X, B, offset=None if off is None else off[d.vet_index])
        yield d, _sigmoid(eta)


def _tidy(cohort: pl.DataFrame, parts: list[pl.DataFrame], value: str) -> pl.DataFrame:
    """Blocks -> one long frame, in cohort order then date then need.

    Ids and need names are attached once, at the end: carrying them per row through a
    10k x 120-day run is most of the memory and none of the work. The sort is what makes
    the table's bytes independent of how many days a block happened to cover.
    """
    if not parts:
        return schema.empty("outcomes").drop("y").with_columns(
            pl.Series(value, [], dtype=pl.Int32 if value == "y" else pl.Float64))
    df = pl.concat(parts).sort("_i", "date", "_k")
    return df.select(
        cohort["veteran_id"].gather(df["_i"]),
        "date",
        pl.col("_k").replace_strict(list(range(_K)), NEEDS, return_dtype=pl.Utf8).alias("need"),
        value,
    )


def _keys(d: design.Design) -> dict[str, pl.Series]:
    n = d.X.shape[0]
    return {
        "_i": pl.Series(np.repeat(d.vet_index, _K), dtype=pl.UInt32),
        "date": pl.Series(np.repeat(d.date, _K), dtype=pl.Date),
        "_k": pl.Series(np.tile(np.arange(_K, dtype=np.int8), n), dtype=pl.Int8),
    }


def _resolve(hazards: pl.DataFrame, dates: Iterable[date] | None,
             truth: Truth | None) -> tuple[list[date], Truth]:
    return (sorted(hazards["date"].unique().to_list()) if dates is None else sorted(dates),
            load() if truth is None else truth)


def rates(cohort: pl.DataFrame, hazards: pl.DataFrame, site_status: pl.DataFrame, *,
          dates: Iterable[date] | None = None, truth: Truth | None = None, seed: int = SEED,
          latent: bool = True, days_per_build: int = DAYS_PER_BUILD) -> pl.DataFrame:
    """The true daily probability per veteran, date and need: `veteran_id, date, need, p`.

    This is the quantity `simulate` draws against. The model never sees it; `eval` uses it
    to ask whether a posterior recovered the world it was given.
    """
    dates, truth = _resolve(hazards, dates, truth)
    parts = [pl.DataFrame({**_keys(d), "p": p.ravel()})
             for d, p in _blocks(cohort, hazards, site_status, dates, truth, seed, latent,
                                 days_per_build)]
    return _tidy(cohort, parts, "p")


def simulate(cohort: pl.DataFrame, hazards: pl.DataFrame, site_status: pl.DataFrame, *,
             dates: Iterable[date] | None = None, truth: Truth | None = None, seed: int = SEED,
             latent: bool = True, days_per_build: int = DAYS_PER_BUILD) -> pl.DataFrame:
    """Bernoulli outcomes for every veteran, date and need -- the `outcomes` contract.

    `hazards` may reach back before `dates`; those extra days feed the lag curves, and its
    first day is the day `days_supply_remaining` was counted on. `latent=False` switches off
    the ZIP and frailty offsets, which is how a test can compare a rate to the intercept.
    """
    dates, truth = _resolve(hazards, dates, truth)
    parts = []
    for d, p in _blocks(cohort, hazards, site_status, dates, truth, seed, latent,
                        days_per_build):
        days = np.unique(d.date)                      # sorted, as design.build sorts dates
        p3 = p.reshape(cohort.height, len(days), _K)  # rows are veteran-major, then date
        u = np.empty_like(p3)
        for t, day in enumerate(days):
            u[:, t, :] = _day_stream(seed, day).random((cohort.height, _K))
        parts.append(pl.DataFrame({**_keys(d),
                                   "y": pl.Series((u < p3).ravel().astype(np.int32),
                                                  dtype=pl.Int32)}))
    return _tidy(cohort, parts, "y")


def quiet_dates(hazards: pl.DataFrame, site_status: pl.DataFrame) -> list[date]:
    """The days on which nothing is happening anywhere -- the off-event baseline.

    A care team doing nothing on a calm June day is the right answer, so these days are
    worth naming rather than averaging away.
    """
    busy = (hazards.group_by("date").agg(
        pl.any_horizontal(
            pl.col("hot_day").any(),
            (pl.col("pm25") > design.PM25_STANDARD).any(),
            pl.col("smoke_alert").any(), pl.col("flood_warning").any(),
            pl.col("flash_flood_emergency").any(), (pl.col("surge_ft") > 0).any(),
            (pl.col("evac_zone_ordered") > 0).any(),
            (pl.col("outage_frac") > QUIET_OUTAGE_FRAC).any(),
            pl.col("mail_delivery_disrupted").any(),
        ).alias("busy")))
    down = site_status.group_by("date").agg(pl.col("site_down").any().alias("busy"))
    return sorted(pl.concat([busy, down]).group_by("date").agg(pl.col("busy").any())
                  .filter(~pl.col("busy"))["date"].to_list())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _summary(out: pl.DataFrame, hazards: pl.DataFrame, sites: pl.DataFrame) -> pl.DataFrame:
    quiet = set(quiet_dates(hazards, sites))
    per_day = out.group_by("need", "date").agg(pl.col("y").mean().alias("rate"))
    return (per_day.group_by("need").agg(
                pl.col("rate").mean().alias("all_days"),
                pl.col("rate").filter(pl.col("date").is_in(sorted(quiet))).mean().alias("quiet"),
                pl.col("rate").max().alias("worst"),
                pl.col("date").sort_by("rate").last().alias("worst_day"))
            .sort(pl.col("need").replace_strict(NEEDS, list(range(_K)))))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, help="simulate only the first N days of the scenario")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--truth", type=Path, default=TRUTH_PATH)
    args = ap.parse_args(argv)

    cohort, hazards = schema.read("cohort"), schema.read("hazards")
    sites = schema.read("site_status")
    truth = load(args.truth)
    dates = sorted(hazards["date"].unique().to_list())[:args.days] if args.days else None

    t0 = time.perf_counter()
    out = simulate(cohort, hazards, sites, dates=dates, truth=truth, seed=args.seed)
    path = schema.write(out, "outcomes")
    secs = time.perf_counter() - t0

    n_days = out["date"].n_unique()
    print(f"  truth {truth.name} (seed {args.seed}): {cohort.height:,} veterans x {n_days} days "
          f"x {len(NEEDS)} needs = {out.height:,} rows, {secs:.1f}s -> {path}")
    quiet = quiet_dates(hazards, sites)
    print(f"  {len(quiet)} of {n_days} days are off-event (no hazard, no closure)")
    for r in _summary(out, hazards, sites).iter_rows(named=True):
        quiet_rate = "n/a" if r["quiet"] is None else f"{r['quiet']:.2%}"
        print(f"  {r['need']:14s} all days {r['all_days']:.2%}  off-event {quiet_rate:>6s}  "
              f"worst {r['worst']:.2%} on {r['worst_day']}")

    down = sites.filter(pl.col("site_down"))
    if down.height:
        closed = cohort.join(down.select("facility_id", "date"), on="facility_id")
        hit = (out.join(closed.select("veteran_id", "date", "ckd_dialysis"),
                        on=["veteran_id", "date"])
                  .filter((pl.col("need") == "treatment_gap") & pl.col("ckd_dialysis")))
        if hit.height:
            station = down["facility_id"].unique().to_list()
            print(f"  dialysis patients of {', '.join(station)} while it is closed: "
                  f"{hit['y'].mean():.1%} treatment-gap days ({hit.height:,} veteran-days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

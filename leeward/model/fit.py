"""`make fit` -- NumPyro NUTS on binomial cells -> `data/posterior.nc` + `report/fit.json`.

    python -m leeward.model.fit                  # the full panel, minus the eval holdout
    python -m leeward.model.fit --toy            # 200 veterans x the 30 days around landfall
    python -m leeward.model.fit --chains 2 --warmup 500 --samples 500

The model is `hazard.py`; this module is the parts that touch the clock, the disk and the
diagnostics. Three things it will not do quietly:

* **It says which rung fitted.** `posterior.nc` carries `model_rung` in its attributes and
  `score.py` refuses a posterior that does not. SPEC 6.0's whole point is that a model which
  silently failed to fit is worse than a simpler one that did.
* **It separates fitted from carried.** At rung 1 sixteen interaction coefficients are
  sampled but held out of the linear predictor, so their posterior is their prior.
  `fit.json` lists both sets by name; nothing may claim the second kind was learned.
* **It fails the run when the sampler failed.** r-hat >= 1.05 or a post-warmup divergence
  exits non-zero, after writing the files, so you can look at what went wrong. The r-hat is
  the rank-normalised split statistic, and the report says whether the run also clears the
  stricter bar in Vehtari et al. (2021) -- r-hat < 1.01 with bulk and tail ESS over 400 --
  because "converged" without a threshold beside it is not a claim (docs/sources.md).

**The last 30 days are not fitted.** `eval/calibration.py` scores its reliability curve on
the last 30 days of the panel, so those days are held out here and every number on the
report screen is out of sample. `--holdout-days 0` fits everything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from leeward import schema
from leeward.model import design, hazard
from leeward.schema import NEEDS

REPORT = schema.ROOT / "report"
FIT_JSON = REPORT / "fit.json"

CHAINS = 4                  # SPEC 6.2: 4 chains x 1,000 warmup x 1,000 samples
WARMUP = 1_000
SAMPLES = 1_000
TARGET_ACCEPT = 0.9
SEED = 0

#: Kept equal to `eval.calibration.HOLDOUT_DAYS`; `tests/test_hazard_toy.py` pins the two
#: together, because a holdout the evaluator does not share is not a holdout.
HOLDOUT_DAYS = 30

RHAT_BAR = 1.05             # SPEC 6.5 acceptance, and what `make fit` exits non-zero on

#: The stricter modern bar, reported next to SPEC's: rank-normalised split-R-hat under 1.01
#: with bulk and tail ESS both over 400 (Vehtari, Gelman, Simpson, Carpenter and Bürkner,
#: *Bayesian Analysis* 16(2), 2021; docs/sources.md). ArviZ's `rhat` already computes the
#: rank-normalised statistic, so this costs one comparison and it is the number a
#: statistician in the room will ask for.
RHAT_BAR_STRICT = 1.01
ESS_BAR_STRICT = 400

#: Diagonal, which is NumPyro's default and what the full fit was validated on. The four
#: heat lags are nearly collinear, so a full-rank metric looks like the obvious cure for a
#: slow fit -- but the geometry is fine: on the full panel the sampler takes **64 leapfrog
#: steps an iteration and saturates the tree depth on 0% of them**. The fit is slow because
#: of volume, not shape: 4,000 iterations x 64 steps is a quarter of a million gradients,
#: each over 106,661 cells. A dense metric would shorten trajectories that are already short
#: and add a 90x90 adaptation to pay for.
#:
#: The flag stays for rung 2, whose extra terms may change that. Note that the one probe run
#: here gave dense only 150 warmup samples to estimate a 90x90 covariance, which is far too
#: few to be a fair test -- so "dense is worse" is not something this lane measured.
DENSE_MASS = False
MAX_TREE_DEPTH = 10         # NumPyro's default, named here because it is a cost ceiling

TOY_VETERANS, TOY_DAYS = 200, 30


# --------------------------------------------------------------------------- #
# Running the sampler
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class Fit:
    """A finished run: the posterior in ArviZ form, what it was fitted on, how it went."""
    idata: object                      # arviz.InferenceData
    cells: hazard.Cells
    blocks: list[hazard.Block]
    diagnostics: dict
    seconds: float

    @property
    def rung(self) -> int:
        return self.cells.rung

    @property
    def converged(self) -> bool:
        """SPEC 6.5's bar: this is what `make fit`'s exit code is about."""
        d = self.diagnostics
        return d["rhat_max"] < RHAT_BAR and d["divergences"] == 0

    @property
    def converged_strict(self) -> bool:
        """Vehtari et al. (2021) as well: R-hat under 1.01, bulk and tail ESS over 400."""
        d = self.diagnostics
        return (self.converged and d["rhat_max"] < RHAT_BAR_STRICT
                and min(d["ess_bulk_min"], d["ess_tail_min"]) > ESS_BAR_STRICT)


def _configure(chains: int, x64: bool) -> str:
    """Put JAX on `chains` host devices so the chains run at once, and pick the precision.

    Both have to happen before JAX brings its backend up, so this is called first. x64 is
    on by default: a float32 sum over 679,000 binomial cells carries enough rounding noise
    into the potential energy to manufacture divergences that are not in the model.

    When the backend is already up -- a pytest process that touched JAX first, say --
    `set_host_device_count` is a no-op and NumPyro quietly drops "parallel" to "sequential",
    four chains one after another. "vectorized" batches the chains into a single JAX call on
    the one device instead, which keeps them running together.
    """
    import numpyro

    numpyro.set_host_device_count(chains)
    if x64:
        numpyro.enable_x64()

    import jax

    return "parallel" if jax.local_device_count() >= chains else "vectorized"


def run(cells: hazard.Cells, *, chains: int = CHAINS, warmup: int = WARMUP,
        samples: int = SAMPLES, seed: int = SEED, target_accept: float = TARGET_ACCEPT,
        rw_sd: float = hazard.RW_SD, scale: float = 1.0, x64: bool = True,
        dense_mass: bool = DENSE_MASS, max_tree_depth: int = MAX_TREE_DEPTH,
        chain_method: str | None = None, progress: bool = False) -> Fit:
    """NUTS on `cells`. Returns the posterior, the diagnostics and the wall clock."""
    if chains < 2:
        raise ValueError(
            f"chains={chains}: r-hat is a comparison between chains and is undefined for "
            "one of them, so a single-chain run cannot answer the question this module "
            "exists to answer. SPEC 6.2 uses four; two is the floor.")
    chain_method = _configure(chains, x64) if chain_method is None else chain_method

    import jax
    import jax.numpy as jnp
    from jax import random
    from numpyro.infer import MCMC, NUTS, init_to_median

    blocks = hazard.blocks(cells.rung, scale=scale)
    args = (jnp.asarray(cells.X), jnp.asarray(cells.n), jnp.asarray(cells.y), blocks)

    kernel = NUTS(hazard.model, init_strategy=init_to_median, target_accept_prob=target_accept,
                  dense_mass=dense_mass, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method=chain_method, progress_bar=progress)
    t0 = time.perf_counter()
    mcmc.run(random.PRNGKey(seed), *args, rw_sd=rw_sd,
             extra_fields=("diverging", "accept_prob", "num_steps", "potential_energy",
                           "energy"))
    # JAX dispatches asynchronously: `mcmc.run` returns once the work is *enqueued*, so a
    # clock stopped here measures dispatch and not sampling. The first version of this
    # module reported 11.3s for a fit that ran for twenty minutes, which is the kind of
    # number that ends up on a slide. Pull the draws back and wait for them.
    draws = jax.block_until_ready(mcmc.get_samples(group_by_chain=True))
    extra = jax.block_until_ready(mcmc.get_extra_fields(group_by_chain=True))
    seconds = time.perf_counter() - t0

    idata = inference_data(draws, extra, blocks)
    diag = diagnostics(idata, blocks, cells, max_tree_depth=max_tree_depth)
    diag["sampler"] = {"chains": chains, "warmup": warmup, "samples": samples,
                       "target_accept": target_accept, "seed": seed,
                       "chain_method": chain_method, "x64": x64,
                       "mass_matrix": "dense" if dense_mass else "diagonal",
                       "max_tree_depth": max_tree_depth,
                       "rw_smoothness_sd": rw_sd, "prior_scale": scale,
                       "init_strategy": "init_to_median"}
    return Fit(idata=idata, cells=cells, blocks=blocks, diagnostics=diag, seconds=seconds)


# --------------------------------------------------------------------------- #
# The posterior file
# --------------------------------------------------------------------------- #

def _dims_and_coords(blocks: list[hazard.Block]) -> tuple[dict, dict]:
    """Every term shares the `need` dimension, and only the two lag curves add one of their
    own. That is what keeps `az.rhat(idata).to_array()` a 2,100-cell array: give each of the
    35 terms its own auto-named dims and xarray broadcasts them all against each other."""
    dims, coords = {}, {"need": list(NEEDS)}
    for b in blocks:
        if b.n_feat == 1:
            dims[b.name] = ["need"]
        else:
            lag = f"{b.name}_lag"
            dims[b.name] = [lag, "need"]
            coords[lag] = list(range(b.n_feat))
    return dims, coords


def inference_data(draws: dict, extra: dict, blocks: list[hazard.Block]):
    """`InferenceData` carrying one variable per design term, named as in `priors.py`.

    Takes the already-fetched draws rather than the `MCMC` object, so that waiting for the
    sampler is something `run()` does on the clock rather than something that happens here
    by accident (see the note about asynchronous dispatch in `run`).

    The `*_raw` sites are dropped: they are the non-centred parameterisation's scaffolding,
    they are recoverable from the coefficients, and every extra variable is another pair of
    dimensions for anything that stacks the posterior into one array.
    """
    import arviz as az

    posterior = {b.name: np.asarray(draws[b.name]) for b in blocks}
    stats = {"diverging": np.asarray(extra["diverging"]),
             "acceptance_rate": np.asarray(extra["accept_prob"]),
             "n_steps": np.asarray(extra["num_steps"]),
             "energy": np.asarray(extra["energy"]),
             "lp": -np.asarray(extra["potential_energy"])}
    dims, coords = _dims_and_coords(blocks)
    return az.from_dict(posterior=posterior, sample_stats=stats, coords=coords, dims=dims)


def _meta(fit: Fit, cells: hazard.Cells) -> dict:
    d = fit.diagnostics
    return {"model_rung": cells.rung,
            "fitted_terms": json.dumps(list(d["fitted_terms"])),
            "carried_terms": json.dumps(list(d["carried_terms"])),
            "n_cells": int(cells.X.shape[0]),
            "n_veteran_days": int(cells.n.sum()),
            "first_date": str(cells.dates[0]),
            "last_date": str(cells.dates[-1]),
            "created": datetime.now(UTC).isoformat(timespec="seconds")}


def _rel(path: Path) -> Path:
    """Repo-relative where that means something, absolute where it does not (a tmp_path)."""
    try:
        return path.relative_to(schema.ROOT)
    except ValueError:
        return path


def write(fit: Fit, *, posterior: Path = hazard.POSTERIOR,
          report: Path = FIT_JSON) -> tuple[Path, Path]:
    """`data/posterior.nc` and `report/fit.json`, in that order."""
    fit.idata.posterior.attrs.update(_meta(fit, fit.cells))
    posterior.parent.mkdir(parents=True, exist_ok=True)
    if posterior.exists():
        posterior.unlink()        # netCDF4 will not overwrite an open-ish file in place
    fit.idata.to_netcdf(str(posterior))

    report.parent.mkdir(parents=True, exist_ok=True)
    payload = {**fit.diagnostics, "runtime_s": round(fit.seconds, 1),
               "posterior": str(_rel(posterior)),
               "generated_at": datetime.now(UTC).isoformat(timespec="seconds")}
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return posterior, report


# --------------------------------------------------------------------------- #
# Diagnostics -- r-hat, divergences, and what the data could not tell us
# --------------------------------------------------------------------------- #

def _cells_by_parameter(stat, blocks: list[hazard.Block]) -> dict[tuple[str, str], float]:
    """An ArviZ per-variable statistic -> {(term, "delta_heat[0] (heat)"): value} over the
    design's supported cells only. A cell outside a term's needs is a structural zero whose
    r-hat is 0/0, and NaN is not a diagnostic."""
    out = {}
    for b in blocks:
        arr = np.atleast_2d(np.asarray(stat[b.name].values))   # (n_feat, K)
        start = design.TERM_SLICE[b.name].start
        for i in range(b.n_feat):
            for k in b.cols:
                v = float(arr[i, k])
                if np.isfinite(v):
                    out[(b.name, f"{design.FEATURES[start + i]} ({NEEDS[k]})")] = v
    return out


def diagnostics(idata, blocks: list[hazard.Block], cells: hazard.Cells, *,
                max_tree_depth: int = MAX_TREE_DEPTH) -> dict:
    """r-hat, divergences, effective sample size, and the features the data is silent on."""
    import arviz as az

    # A coefficient outside its term's needs is written as a constant zero, and r-hat of a
    # constant is 0/0. `_cells_by_parameter` drops those cells; this keeps numpy from
    # narrating it. Anything else invalid still shows up as a NaN we would then not see.
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = _cells_by_parameter(az.rhat(idata), blocks)          # rank-normalised split
        ess = _cells_by_parameter(az.ess(idata, method="bulk"), blocks)
        ess_tail = _cells_by_parameter(az.ess(idata, method="tail"), blocks)
    fitted = [b.name for b in blocks if b.active]
    fitted_rhat = [v for (term, _), v in rhat.items() if term in fitted]

    diverging = np.asarray(idata.sample_stats["diverging"].values)
    label = lambda kv: kv[0][1]      # noqa: E731 -- (term, parameter) -> parameter
    worst = sorted(rhat.items(), key=lambda kv: -kv[1])[:5]
    thin = sorted(ess.items(), key=lambda kv: kv[1])[:5]

    x = cells.X
    flat = [n for j, n in enumerate(cells.names) if np.ptp(x[:, j]) == 0 and n != "alpha"]

    return {
        "model_rung": cells.rung,
        "rhat_max": max(rhat.values()),
        "rhat_max_fitted": max(fitted_rhat) if fitted_rhat else None,
        "divergences": int(diverging.sum()),
        "divergences_by_chain": [int(c) for c in diverging.sum(axis=1)],
        "ess_bulk_min": min(ess.values()),
        "ess_tail_min": min(ess_tail.values()),
        "bars": {"rhat_spec": RHAT_BAR, "rhat_strict": RHAT_BAR_STRICT,
                 "ess_strict": ESS_BAR_STRICT,
                 "rhat_statistic": "rank-normalised split R-hat (Vehtari et al. 2021)"},
        "acceptance_rate_mean": float(
            np.asarray(idata.sample_stats["acceptance_rate"].values).mean()),
        # Leapfrog steps per iteration is what a slow fit is made of, and the one number
        # that was missing when the first full run sampled for half an hour in silence.
        # Saturating the tree depth means the geometry, not the data, is the cost.
        "leapfrog_steps_mean": float(np.asarray(idata.sample_stats["n_steps"].values).mean()),
        "tree_depth_saturated": float(
            (np.asarray(idata.sample_stats["n_steps"].values) >= 2 ** max_tree_depth - 1).mean()),
        "worst_rhat": [{"parameter": label(kv), "rhat": round(kv[1], 4)} for kv in worst],
        "lowest_ess_bulk": [{"parameter": label(kv), "ess_bulk": round(kv[1], 1)}
                            for kv in thin],
        "fitted_terms": fitted,
        "carried_terms": [b.name for b in blocks if not b.active],
        "flat_features": flat,
        "data": {
            "veteran_days": int(cells.n.sum()),
            "cells": int(cells.X.shape[0]),
            "reduction": round(cells.reduction, 1),
            "days": len(cells.dates),
            "first_date": str(cells.dates[0]),
            "last_date": str(cells.dates[-1]),
            "events_per_need": {n: int(cells.y[:, k].sum()) for k, n in enumerate(NEEDS)},
        },
    }


def _verdict(fit: Fit) -> str:
    """Which bars the run cleared, in a sentence that is never generous about it."""
    if not fit.converged:
        return (f"does NOT clear SPEC's r-hat < {RHAT_BAR} with zero divergences. "
                f"This run is not a fit you may quote.")
    if fit.converged_strict:
        return (f"clears SPEC's r-hat < {RHAT_BAR} and Vehtari et al.'s < {RHAT_BAR_STRICT} "
                f"with bulk and tail ESS > {ESS_BAR_STRICT}")
    return (f"clears SPEC's r-hat < {RHAT_BAR}, but not the stricter Vehtari et al. bar "
            f"(< {RHAT_BAR_STRICT} with bulk and tail ESS > {ESS_BAR_STRICT})")


def summary(fit: Fit) -> str:
    """The three sentences you say on stage, and the ones you say if it went wrong."""
    d, c = fit.diagnostics, fit.cells
    lines = [
        f"rung {fit.rung}: {c.n_rows:,} veteran-days -> {c.X.shape[0]:,} binomial cells "
        f"({c.reduction:.1f}x) over {len(c.dates)} days, {c.dates[0]} .. {c.dates[-1]}",
        f"  {d['sampler']['chains']} chains x {d['sampler']['warmup']}+"
        f"{d['sampler']['samples']} ({d['sampler']['chain_method']}, "
        f"{'float64' if d['sampler']['x64'] else 'float32'}) in {fit.seconds:.1f}s "
        f"including JIT compilation",
        f"  r-hat max {d['rhat_max']:.4f} · {d['divergences']} divergences · ESS min "
        f"{d['ess_bulk_min']:,.0f} bulk / {d['ess_tail_min']:,.0f} tail · "
        f"accept {d['acceptance_rate_mean']:.2f}",
        f"  {d['sampler']['mass_matrix']} metric · {d['leapfrog_steps_mean']:.0f} leapfrog "
        f"steps an iteration · tree depth saturated on {d['tree_depth_saturated']:.1%} of them",
        "  " + _verdict(fit),
        f"  fitted {len(d['fitted_terms'])} terms; {len(d['carried_terms'])} carried at "
        f"their priors and marked as such: {', '.join(d['carried_terms'][:3])}"
        f"{' ...' if len(d['carried_terms']) > 3 else ''}",
    ]
    events = " ".join(f"{n} {v:,}" for n, v in d["data"]["events_per_need"].items())
    lines.append(f"  events the fit saw: {events}")
    if d["flat_features"]:
        lines.append(f"  no variation in this window, so these stayed at their priors: "
                     f"{', '.join(d['flat_features'])}")
    if not fit.converged:
        lines.append(f"  NOT CONVERGED: worst r-hat {d['worst_rhat'][0]['parameter']} "
                     f"{d['worst_rhat'][0]['rhat']}, {d['divergences']} divergences. "
                     f"Say so, or refit with more warmup.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

#: What a hazard day looks like, one expression per rung-1 hazard term. A window that
#: contains none of these fits five intercepts and calls it a model.
_HAZARD_KINDS = {
    "hot": pl.col("hot_day"),
    "smoke": pl.col("smoke_alert") | (pl.col("pm25") > design.PM25_STANDARD),
    "flood": (pl.col("flood_warning") | pl.col("flash_flood_emergency")
              | pl.col("floodnet_trip") | (pl.col("surge_ft") > 0)),
    "evac": pl.col("evac_zone_ordered") > 0,
    "outage": pl.col("outage_frac") > 0.5,
    "mail": pl.col("mail_delivery_disrupted"),
}


def _activity(dates: list[date], hazards: pl.DataFrame,
              sites: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(T, kinds) which hazards fired on each date, and (T,) how many ZIP-days they cost."""
    per_day = (hazards.filter(pl.col("date").is_in(dates)).group_by("date")
               .agg(*[e.sum().alias(k) for k, e in _HAZARD_KINDS.items()])
               .join(sites.group_by("date").agg(pl.col("site_down").sum().alias("site_down")),
                     on="date", how="left")
               .sort("date"))
    kinds = list(_HAZARD_KINDS) + ["site_down"]
    counts = per_day.select(kinds).fill_null(0).to_numpy().astype(float)
    return counts > 0, counts.sum(axis=1)


def toy_window(dates: list[date], hazards: pl.DataFrame, sites: pl.DataFrame,
               n_days: int = TOY_DAYS) -> list[date]:
    """The `n_days` window that exercises the most of the hazard terms.

    A toy fit on a calm fortnight samples beautifully and proves nothing: every hazard
    coefficient comes back sitting on its prior, and the one thing the toy fit is for is
    catching a term that does not reach the data. Windows are ranked by how many *kinds* of
    hazard they contain and then by how many ZIP-days those cost, so on the Sandy scenario
    this is the month around landfall -- surge, evacuation, outage, closure and the heat
    wave three days later.
    """
    if len(dates) <= n_days:
        return dates
    fired, cost = _activity(dates, hazards, sites)
    starts = range(len(dates) - n_days + 1)
    best = max(starts, key=lambda s: (int(fired[s:s + n_days].any(axis=0).sum()),
                                      float(cost[s:s + n_days].sum()), -s))
    return dates[best:best + n_days]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rung", type=int, default=hazard.RUNG,
                    help="1 is what this lane tested; 2 adds the interactions (SPEC 6.0)")
    ap.add_argument("--toy", action="store_true",
                    help=f"{TOY_VETERANS} veterans x the {TOY_DAYS} days that exercise the "
                         f"most hazard terms")
    ap.add_argument("--veterans", type=int, help="fit on the first N veterans only")
    ap.add_argument("--days", type=int, help="fit on the first N days of the window")
    ap.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS,
                    help="days at the end left unfitted for eval/calibration.py")
    ap.add_argument("--chains", type=int, default=CHAINS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--samples", type=int, default=SAMPLES)
    ap.add_argument("--target-accept", type=float, default=TARGET_ACCEPT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--scale", type=float, default=1.0, help="prior-scale multiplier")
    ap.add_argument("--no-x64", action="store_true", help="sample in float32 (faster, noisier)")
    ap.add_argument("--dense-mass", action=argparse.BooleanOptionalAction, default=DENSE_MASS,
                    help="full-rank mass matrix; the heat lags are nearly collinear")
    ap.add_argument("--max-tree-depth", type=int, default=MAX_TREE_DEPTH)
    # A per-chain progress bar is worth having at a terminal and is noise in a log, so it
    # follows the terminal rather than needing to be remembered.
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction,
                    default=sys.stdout.isatty(), help="per-chain progress bars")
    ap.add_argument("--out", type=Path, default=hazard.POSTERIOR)
    ap.add_argument("--report", type=Path, default=FIT_JSON)
    args = ap.parse_args(argv)
    if args.chains < 2:
        ap.error("--chains must be at least 2: r-hat compares chains and is undefined for "
                 "one. SPEC 6.2 uses 4.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rung != hazard.RUNG:
        print(f"! rung {args.rung} is not covered by this lane's tests; SPEC 6.0 makes it "
              f"the next task. Everything it needs is hazard.LADDER[{args.rung}].")

    cohort, hazards = schema.read("cohort"), schema.read("hazards")
    sites, outcomes = schema.read("site_status"), schema.read("outcomes")

    dates = sorted(set(hazards["date"].unique().to_list())
                   & set(outcomes["date"].unique().to_list()))
    if not dates:
        print("hazards and outcomes share no dates; run `make cohort` first")
        return 1
    holdout = min(max(args.holdout_days, 0), len(dates))
    held, dates = dates[len(dates) - holdout:], dates[:len(dates) - holdout]

    if args.toy:
        cohort = cohort.head(args.veterans or TOY_VETERANS)
        dates = toy_window(dates, hazards, sites, args.days or TOY_DAYS)
    else:
        if args.veterans:
            cohort = cohort.head(args.veterans)
        if args.days:
            dates = dates[:args.days]
    if not dates:
        print("no days left to fit; --holdout-days is longer than the panel")
        return 1

    cells = hazard.cells(cohort, hazards, sites, outcomes, dates=dates, rung=args.rung)
    # Say what is about to happen before it happens. The full panel samples for tens of
    # minutes, and a target that prints nothing until it is finished is a target nobody can
    # tell from a hung one -- which is how the first full run got killed.
    print(f"rung {args.rung}: {cells.n_rows:,} veteran-days -> {cells.X.shape[0]:,} cells "
          f"({cells.reduction:.1f}x) over {len(cells.dates)} days, "
          f"{cells.dates[0]} .. {cells.dates[-1]}")
    print(f"  sampling {args.chains} chains x {args.warmup}+{args.samples} over "
          f"{len(cells.names)} features x {len(NEEDS)} needs; minutes, not seconds",
          flush=True)
    fit = run(cells, chains=args.chains, warmup=args.warmup, samples=args.samples,
              seed=args.seed, target_accept=args.target_accept, scale=args.scale,
              x64=not args.no_x64, dense_mass=args.dense_mass,
              max_tree_depth=args.max_tree_depth, progress=args.progress)
    fit.diagnostics["data"]["veterans"] = cohort.height
    fit.diagnostics["data"]["holdout_days"] = len(held)
    fit.diagnostics["data"]["holdout_first_date"] = str(held[0]) if held else None
    fit.diagnostics["validated_rung"] = hazard.RUNG

    nc, js = write(fit, posterior=args.out, report=args.report)
    print(summary(fit))
    if held:
        print(f"  {len(held)} days held out of the fit for eval/calibration.py: "
              f"{held[0]} .. {held[-1]}")
    print(f"  -> {_rel(nc)}, {_rel(js)}")
    return 0 if fit.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())

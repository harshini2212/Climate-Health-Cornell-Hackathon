"""Rung 1 of the ladder (docs/SPEC.md §6.0): pooled NUTS on binomial cells.

Rung 0 multiplies `design.py` by draws from `priors.py`. This rung keeps the same design and
the same coefficient names and lets the data move them:

    cells = hazard.cells(cohort, hazards, site_status, outcomes)   # sufficient statistics
    mcmc  = fit.run(cells)                                          # NumPyro NUTS
    -> data/posterior.nc, variable names matching priors.py

**What rung 1 contains** is the main-effect block: `alpha`, the ten health betas, the two
social terms, the 4-lag heat curve, the 3-lag PM2.5 curve, and the flood, evacuation, outage
and SiteDown hazard terms -- 24 of the design's 40 features. No ICAR, no latent burn-pit
dose, no interactions (SPEC §6.0; the TODOs are at the bottom of this file).

**Every term is still sampled, and every term is still written.** The sixteen interaction
coefficients are declared in the model but their design columns are held out of the linear
predictor, so their posterior is exactly their prior -- which is the honest thing for a
coefficient no likelihood has touched, and it keeps `data/posterior.nc` carrying all 35
terms as `eval/recovery.py` requires. `fit.json` lists which terms were fitted and which
were carried, and nothing downstream may claim the second kind was learned from data.

**A fitted main effect is marginal over the terms that are missing.** Nearly every day in
August is a hot day, so the six heat interactions the likelihood cannot see are close to
constant, and `alpha (heat)` absorbs them: it comes back near -4.4 where the simulator
generated -6.0. That is not a bug, it is what fitting a marginal model means, and it is the
measured argument for rung 2. It does have one consequence downstream, enforced in
`score.py`: **predict with the terms you fitted.** Adding prior draws for the interactions
back on top of main effects that already contain them would count them twice.

Binomial cells
--------------
The likelihood depends on a veteran-day only through its row of `X`, so two veterans with
the same covariates under the same weather are one cell, whatever ZIP or day they came
from, and the five needs share that cell because they share the design row. Over the whole
120-day panel, 1,200,000 veteran-days collapse to 135,743 cells -- 8.8x fewer, and the same
8.8x on the 6,000,000 Bernoulli terms, since a cell still carries one binomial term per
need. Grouping on the exact float row keeps the likelihood *identical* rather than
approximately equal, and `tests/test_hazard_toy.py` asserts exactly that against the
Bernoulli sum over the panel.

Grouping is on the **active** columns only: at rung 1 two rows that differ solely in an
interaction feature are one cell, because nothing in the likelihood can tell them apart.

What this rung does not model
-----------------------------
`cohort/simulate.py` draws outcomes with a per-ZIP effect (sigma 0.4) and a per-veteran
frailty (sigma 0.5) that this model has no term for. They do not vanish: they make the cells
overdispersed relative to a binomial and pull the fitted coefficients toward zero, the usual
attenuation of a marginal model. Rung 3's ICAR prior is where the ZIP half gets modelled.
Until then it is a known bias, reported rather than hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from leeward import schema
from leeward.model import design, priors
from leeward.schema import NEEDS

POSTERIOR = schema.DATA / "posterior.nc"

RUNG = 1

#: SPEC 6.3 gives the lag curves a random-walk smoothness prior on top of their Normal:
#: consecutive lags of one curve are one day apart and should not be free to zigzag. Heat on
#: day t-1 and day t-2 are nearly collinear, so this is also what keeps the curve identified.
RW_SD = 0.2

#: Interaction terms are named for the theta of SPEC 6.3; everything else is a main effect.
#: Rung 2's whole job is to move that line, so the ladder is written as term lists, not ifs.
INTERACTIONS: tuple[str, ...] = tuple(t.name for t in design.TERMS
                                      if t.name.startswith("theta_"))
MAIN_EFFECTS: tuple[str, ...] = tuple(t.name for t in design.TERMS
                                      if not t.name.startswith("theta_"))

#: Which terms enter the linear predictor at each rung. A term that is not here is still
#: sampled, and still written to posterior.nc, at its prior.
LADDER: dict[int, tuple[str, ...]] = {
    0: (),                                  # prior-only, no likelihood -- score_prior.py
    1: MAIN_EFFECTS,
    2: MAIN_EFFECTS + INTERACTIONS,         # TODO(rung 2): the six med interactions and psi
}
# TODO(rung 3): ICAR phi per ZIP per need (needs the MODZCTA adjacency graph, which nothing
# builds yet) and the latent burn-pit dose with its pact_presumptive measurement model.
# Both add sampled sites that design.py has no column for, so `Cells` grows a ZIP index and
# `model()` grows two blocks; nothing else in this file changes.

assert set(MAIN_EFFECTS) | set(INTERACTIONS) == set(design.TERM_BY_NAME), (
    "every design term must land in exactly one half of the ladder")


# --------------------------------------------------------------------------- #
# Which coefficients the likelihood can see
# --------------------------------------------------------------------------- #

def active_features(rung: int = RUNG) -> np.ndarray:
    """(P,) True for the design columns whose coefficients enter the linear predictor."""
    if rung not in LADDER:
        raise ValueError(f"rung {rung} is not on the ladder; SPEC 6.0 has {sorted(LADDER)}")
    mask = np.zeros(design.P, dtype=bool)
    for name in LADDER[rung]:
        mask[design.TERM_SLICE[name]] = True
    return mask


def feature_names(rung: int = RUNG) -> list[str]:
    return [f for f, on in zip(design.FEATURES, active_features(rung), strict=True) if on]


# --------------------------------------------------------------------------- #
# Sufficient statistics
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class Cells:
    """Binomial cells: one row per distinct active design row, five needs wide.

    `y[c, k]` of `n[c]` veteran-days in cell `c` had need `k`. `X` holds only the active
    columns, in `names` order, which is `design.FEATURES` filtered by the rung.
    """
    X: np.ndarray            # (C, P_active) float64
    n: np.ndarray            # (C,) int32 -- veteran-days in the cell
    y: np.ndarray            # (C, K) int32 -- events, one column per need in NEEDS order
    names: tuple[str, ...]   # the active feature names, matching X's columns
    rung: int
    n_rows: int              # veteran-days that went in, for the reduction ratio
    dates: tuple[date, ...]

    @property
    def reduction(self) -> float:
        return self.n_rows / self.X.shape[0]

    def __repr__(self) -> str:
        return (f"Cells(rung={self.rung}, {self.X.shape[0]:,} cells from {self.n_rows:,} "
                f"veteran-days, {self.reduction:.1f}x, {len(self.names)} features)")


def _wide_outcomes(outcomes: pl.DataFrame) -> pl.DataFrame:
    """`veteran_id, date, y_<need> x 5`. One pivot for the whole run, not one per block."""
    missing = set(NEEDS) - set(outcomes["need"].unique().to_list())
    if missing:
        raise ValueError(f"outcomes has no rows for need(s) {sorted(missing)}")
    wide = outcomes.pivot(on="need", index=["veteran_id", "date"], values="y")
    return wide.select("veteran_id", "date",
                       *[pl.col(n).cast(pl.Int32).alias(f"y_{n}") for n in NEEDS])


def _block_outcomes(d: design.Design, wide: pl.DataFrame) -> np.ndarray:
    """(N, K) outcomes lined up with the design's rows, in the design's order."""
    keys = pl.DataFrame({"veteran_id": pl.Series(d.veteran_id, dtype=pl.Utf8),
                         "date": pl.Series(d.date).cast(pl.Date)})
    rows = keys.join(wide, on=["veteran_id", "date"], how="left", maintain_order="left")
    gap = rows.filter(pl.col(f"y_{NEEDS[0]}").is_null()).select("veteran_id", "date")
    if gap.height:
        sample = ", ".join(f"{v} {dt}" for v, dt in gap.head(3).rows())
        raise ValueError(
            f"{gap.height:,} veteran-days have no outcome row (e.g. {sample}). A missing "
            "outcome must not read as nothing happened -- run `make cohort` for this window.")
    return rows.select([f"y_{n}" for n in NEEDS]).to_numpy().astype(np.int32, copy=False)


def cells(cohort: pl.DataFrame, hazards: pl.DataFrame, site_status: pl.DataFrame,
          outcomes: pl.DataFrame, *, dates: list[date] | None = None, rung: int = RUNG,
          ref_date: date | None = None, days_per_build: int = 7) -> Cells:
    """Group the panel into binomial cells for `rung`.

    The design matrix is built a week at a time, as `simulate.py` and `score_prior.py` do,
    so a 10,000 x 120 panel never exists in memory at once. Cells are summed across blocks
    and sorted at the end, so the same panel always produces the same cell order and the
    same seed therefore produces the same chain.
    """
    dates = sorted(hazards["date"].unique().to_list()) if dates is None else sorted(dates)
    ref = hazards["date"].min() if ref_date is None else ref_date
    names = feature_names(rung)
    if not names:
        raise ValueError(f"rung {rung} has no fitted terms; it is prior-only (score_prior.py)")
    active = active_features(rung)
    y_cols = [f"y_{n}" for n in NEEDS]

    wide = _wide_outcomes(outcomes.filter(pl.col("date").is_in(dates)))
    parts = []
    for i in range(0, len(dates), days_per_build):
        d = design.build(cohort, hazards, site_status,
                         dates=dates[i:i + days_per_build], ref_date=ref)
        Y = _block_outcomes(d, wide)
        X = d.X[:, active]
        frame = pl.DataFrame({**{f: X[:, j] for j, f in enumerate(names)},
                              **{c: Y[:, k] for k, c in enumerate(y_cols)}})
        parts.append(frame.group_by(names).agg(pl.len().alias("n"),
                                               *[pl.col(c).sum() for c in y_cols]))

    grouped = (pl.concat(parts).group_by(names)
               .agg(pl.col("n").sum(), *[pl.col(c).sum() for c in y_cols])
               .sort(names))
    return Cells(X=grouped.select(names).to_numpy().astype(np.float64, copy=False),
                 n=grouped["n"].to_numpy().astype(np.int32),
                 y=grouped.select(y_cols).to_numpy().astype(np.int32),
                 names=tuple(names), rung=rung, n_rows=cohort.height * len(dates),
                 dates=tuple(dates))


# --------------------------------------------------------------------------- #
# The coefficient blocks -- priors.py, laid out the way the sampler wants them
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class Block:
    """One term's coefficients: what to sample, where it lands in B, whether it is fitted."""
    name: str
    cols: tuple[int, ...]    # the needs this term acts on, as indices into NEEDS
    mean: np.ndarray         # (n_feat, n_params) prior means, in log-odds
    sd: np.ndarray           # (n_params,) prior sds
    shared: bool             # one parameter across the needs (SPEC 6.3's shared thetas)
    lagged: bool             # a distributed-lag curve, so the RW smoothness prior applies
    active: bool             # does it enter the linear predictor at this rung

    @property
    def n_feat(self) -> int:
        return self.mean.shape[0]


def blocks(rung: int = RUNG, *, scale: float = 1.0) -> list[Block]:
    """One `Block` per design term, in design order. `scale` is the prior-scale multiplier."""
    on = set(LADDER[rung])
    out = []
    for term in design.TERMS:
        spec = priors.per_need(term.name)
        shared = isinstance(priors.PRIORS[term.name], priors.Prior)
        cols = tuple(NEEDS.index(n) for n in term.needs)
        per = [spec[term.needs[0]]] if shared else [spec[n] for n in term.needs]
        mean = np.stack([np.atleast_1d(np.asarray(p.mean, dtype=np.float64)) for p in per],
                        axis=1)                                   # (n_feat, n_params)
        sd = np.array([p.sd for p in per], dtype=np.float64) * scale
        out.append(Block(name=term.name, cols=cols, mean=mean, sd=sd, shared=shared,
                         lagged=term.lagged, active=term.name in on))
    return out


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

def model(X, n, y, blocks_: list[Block], *, rw_sd: float = RW_SD) -> None:
    """Binomial cells with a non-centred Normal on every coefficient (SPEC 6.2).

    `X` is (C, P_active), `n` is (C,), `y` is (C, K). Every block is sampled; only the
    active ones reach `eta`, so an inactive coefficient's posterior is its prior.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    K = design.K
    fitted = []
    for b in blocks_:
        raw = numpyro.sample(f"{b.name}_raw",
                             dist.Normal(0.0, 1.0).expand(list(b.mean.shape)).to_event(2))
        coef = b.mean + b.sd * raw                                 # (n_feat, n_params)
        if b.lagged and rw_sd:
            # RW(rw_sd) on the curve itself, not on the raw: the smoothness is a statement
            # about log-odds per day, and it is what separates four collinear heat lags.
            step = jnp.diff(coef, axis=0)
            numpyro.factor(f"{b.name}_rw", -0.5 * jnp.sum(step ** 2) / rw_sd ** 2)
        spread = jnp.broadcast_to(coef, (b.n_feat, len(b.cols)))   # shared -> every need
        block = jnp.zeros((b.n_feat, K)).at[:, list(b.cols)].set(spread)
        # Single-feature terms are written (need,) and lag curves (lag, need); both shapes
        # are what eval/recovery.py accepts, and sharing the `need` dim is what keeps
        # `az.rhat(idata).to_array()` from broadcasting 70 one-off dims against each other.
        numpyro.deterministic(b.name, block[0] if b.n_feat == 1 else block)
        if b.active:
            fitted.append(block)

    eta = X @ jnp.concatenate(fitted, axis=0)                      # (C, K) log-odds
    numpyro.sample("y", dist.BinomialLogits(logits=eta, total_count=n[:, None]), obs=y)


def _log_expit(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -x)


def log_likelihood(cells_: Cells, B: np.ndarray) -> float:
    """log p(y | B) for the cells, at a full (P, K) coefficient matrix.

    Up to the binomial coefficient, which is constant in B and so changes no posterior --
    which is exactly why this is comparable, term for term, with the Bernoulli
    log-likelihood of the panel the cells were grouped from.
    """
    eta = cells_.X @ B[active_features(cells_.rung)]
    n, y = cells_.n[:, None], cells_.y
    return float((y * _log_expit(eta) + (n - y) * _log_expit(-eta)).sum())


# --------------------------------------------------------------------------- #
# The posterior contract -- what fit.py writes and score.py reads back
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class Posterior:
    """`data/posterior.nc`, read back into the (D, P, K) layout `design.py` works in."""
    B: np.ndarray            # (draws, P, K); structural zeros stay exactly zero
    rung: int
    fitted: tuple[str, ...]  # terms the likelihood saw
    carried: tuple[str, ...] # terms whose posterior is their prior
    attrs: dict

    @property
    def n_draws(self) -> int:
        return self.B.shape[0]


def read_posterior(path: str | Path = POSTERIOR) -> Posterior:
    """`posterior.nc` -> (draws, P, K), plus the rung that wrote it.

    Strict on purpose: a term the file does not carry, or carries at the wrong shape, is an
    error rather than a zero. Zeroing it would delete a term from the model on the way out
    and no downstream number would look wrong.
    """
    import arviz as az

    idata = az.from_netcdf(str(path))
    post = idata.posterior
    B = None
    for term in design.TERMS:
        if term.name not in post:
            raise ValueError(f"{path}: no variable {term.name!r}; every rung writes every "
                             f"term. Present: {sorted(post.data_vars)}")
        arr = np.asarray(post[term.name].values)
        n_feat = len(term.exprs)
        want = (n_feat, design.K) if n_feat > 1 else (design.K,)
        if arr.shape[2:] != want:
            raise ValueError(f"{path}: {term.name} has trailing shape {arr.shape[2:]}, "
                             f"expected {want}")
        draws = arr.shape[0] * arr.shape[1]
        if B is None:
            B = np.zeros((draws, design.P, design.K))
        elif draws != B.shape[0]:
            raise ValueError(f"{path}: {term.name} has {draws} draws, earlier terms "
                             f"had {B.shape[0]}")
        B[:, design.TERM_SLICE[term.name], :] = arr.reshape(draws, n_feat, design.K)
    B[:, ~design.SUPPORT] = 0.0

    attrs = dict(post.attrs)
    if "model_rung" not in attrs:
        raise ValueError(f"{path} does not say which rung wrote it. Re-run `make fit`; a "
                         "posterior that cannot name its rung cannot be shown on stage.")
    return Posterior(B=B, rung=int(attrs["model_rung"]),
                     fitted=tuple(json.loads(attrs.get("fitted_terms", "[]"))),
                     carried=tuple(json.loads(attrs.get("carried_terms", "[]"))),
                     attrs=attrs)

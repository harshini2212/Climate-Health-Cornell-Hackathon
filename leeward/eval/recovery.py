"""Recovery -- do the intervals contain the coefficients the world was generated from? SPEC §11.

`data/truth.json` holds the log-odds the simulator drew outcomes from. This module puts each
of them next to a 90 percent interval and says, per parameter, whether the interval contains
it. SPEC §11's bar is 90 percent of parameters covered.

**What the interval is depends on the rung, and the module says which.** At rung 0 there is
no MCMC, so the interval comes from `priors.draw`: the honest reading is then "do the priors
bracket the truth", which is a much weaker claim than parameter recovery and has to be
labelled as one. Every row carries a `source` column, the chart says it in its subtitle, and
`report.json` says it by way of `model_rung: 0` with a null `rhat_max`. From rung 1 the same
code reads `data/posterior.nc` and the claim becomes recovery proper.

Both sides go through `leeward.model.design`: the truth through `coef_matrix`, the draws
into the same (P, K) layout. That is deliberate -- a renamed term then fails loudly in one
place instead of quietly becoming a zero on both sides at once and "recovering" perfectly.

    python -m leeward.eval.recovery     # -> report/recovery.{csv,html}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from leeward import schema
from leeward.model import design, priors
from leeward.schema import NEEDS

REPORT = schema.ROOT / "report"
TRUTH = schema.DATA / "truth.json"
POSTERIOR = schema.DATA / "posterior.nc"

N_DRAWS = 4000                    # rung-0 prior draws; enough for a stable 5th percentile
SEED = 0
LO_PCT, HI_PCT = 5.0, 95.0        # the 90 percent interval
COVERAGE_BAR = 0.90               # SPEC §11

RECOVERY_SCHEMA = {"parameter": pl.Utf8, "term": pl.Utf8, "need": pl.Utf8,
                   "truth": pl.Float64, "post_mean": pl.Float64, "lo90": pl.Float64,
                   "hi90": pl.Float64, "covered": pl.Boolean, "source": pl.Utf8}

#: The (P, K) block shape `fit.py` should write per term, and what this module will accept.
SHAPE_NOTE = ("(chain, draw, n_features, n_needs); (chain, draw, n_needs) when the term has "
              "one feature; (chain, draw, n_features) when it acts on one need")


# --------------------------------------------------------------------------- #
# The two sides
# --------------------------------------------------------------------------- #

def truth_coefficients(path: Path = TRUTH) -> np.ndarray:
    """(P, K) generating coefficients, read through the design the simulator used."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return design.coef_matrix(raw["coefficients"])


def parameter_name(p: int, k: int) -> str:
    """The label for one coefficient cell, e.g. `delta_heat[0] (heat)`."""
    return f"{design.FEATURES[p]} ({NEEDS[k]})"


def posterior_draws(idata) -> np.ndarray:
    """An ArviZ `InferenceData` -> (D, P, K), draws first, structural zeros left at zero.

    Variable names match the keys in `priors.py` (and so the terms in `design.py`), which is
    the contract every rung writes. A missing term or an unexpected shape raises: filling it
    with zeros would delete a term from one side of the comparison and call it recovered.
    """
    post = idata.posterior
    B: np.ndarray | None = None
    for term in design.TERMS:
        if term.name not in post:
            raise ValueError(
                f"{term.name} is in design.py but not in the posterior; every rung writes "
                f"every term. Variables present: {sorted(post.data_vars)}")
        arr = np.asarray(post[term.name].values)
        if arr.ndim < 2:
            raise ValueError(f"{term.name}: expected leading (chain, draw); got {arr.shape}")
        D = arr.shape[0] * arr.shape[1]
        if B is None:
            B = np.zeros((D, design.P, design.K))
        elif D != B.shape[0]:
            raise ValueError(f"{term.name}: {D} draws, but earlier terms had {B.shape[0]}")

        rows = design.TERM_SLICE[term.name]
        n_feat = rows.stop - rows.start
        cols = [NEEDS.index(n) for n in term.needs]
        tail, flat = arr.shape[2:], arr.reshape(D, *arr.shape[2:])
        if tail == (n_feat, design.K):
            B[:, rows, :] = flat
        elif n_feat == 1 and tail == (design.K,):
            B[:, rows.start, :] = flat
        elif len(cols) == 1 and tail == (n_feat,):
            B[:, rows, cols[0]] = flat
        elif n_feat == 1 and tail == ():
            B[:, rows.start, cols] = flat[:, None]
        else:
            raise ValueError(
                f"{term.name}: trailing shape {tail} fits none of {SHAPE_NOTE}; the term has "
                f"{n_feat} feature(s) and acts on {list(term.needs)}")
    if B is None:
        raise ValueError("design.py has no terms")
    B[:, ~design.SUPPORT] = 0.0       # anything a term wrote outside its own needs
    return B


def coefficient_draws(*, n_draws: int = N_DRAWS, seed: int = SEED, scale: float = 1.0,
                      posterior: Path | None = POSTERIOR) -> tuple[np.ndarray, str]:
    """(draws, source): `data/posterior.nc` when it exists, otherwise the rung-0 priors.

    `source` is "posterior" or "prior", and the caller is expected to carry it into the
    table -- a coverage number that does not say which one it measured is a claim the build
    has not earned.
    """
    if posterior is not None and Path(posterior).exists():
        import arviz as az
        return posterior_draws(az.from_netcdf(Path(posterior))), "posterior"
    return priors.draw(n_draws, seed=seed, scale=scale), "prior"


def diagnostics(posterior: Path | None = POSTERIOR) -> tuple[float | None, int | None]:
    """(max r-hat, divergence count) from `posterior.nc`, or (None, None) at a rung with
    no MCMC. Never a placeholder number: "we did not sample" and "it sampled perfectly"
    have to look different on the report screen."""
    if posterior is None or not Path(posterior).exists():
        return None, None
    import arviz as az
    idata = az.from_netcdf(Path(posterior))
    rhat = float(np.nanmax(az.rhat(idata).to_array().values))
    diverging = getattr(idata, "sample_stats", None)
    n_div = int(np.asarray(diverging["diverging"].values).sum()) if (
        diverging is not None and "diverging" in diverging) else None
    return rhat, n_div


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #

def recover(truth: np.ndarray, draws: np.ndarray, *, source: str = "prior") -> pl.DataFrame:
    """One row per coefficient the design supports: truth, posterior mean, 90 pct interval.

    `source` is what `coefficient_draws` returned. Uncovered rows stay in the table -- they
    are the rows worth reading.
    """
    if truth.shape != (design.P, design.K):
        raise ValueError(f"truth must be {(design.P, design.K)}; got {truth.shape}")
    if draws.ndim != 3 or draws.shape[1:] != (design.P, design.K):
        raise ValueError(f"draws must be (D, {design.P}, {design.K}); got {draws.shape}")

    mean = draws.mean(axis=0)
    lo = np.percentile(draws, LO_PCT, axis=0)
    hi = np.percentile(draws, HI_PCT, axis=0)
    cells = [(p, k) for p in range(design.P) for k in range(design.K) if design.SUPPORT[p, k]]
    return pl.DataFrame(
        [{"parameter": parameter_name(p, k),
          "term": design.FEATURES[p], "need": NEEDS[k],
          "truth": float(truth[p, k]), "post_mean": float(mean[p, k]),
          "lo90": float(lo[p, k]), "hi90": float(hi[p, k]),
          "covered": bool(lo[p, k] <= truth[p, k] <= hi[p, k]), "source": source}
         for p, k in cells],
        schema=RECOVERY_SCHEMA)


def coverage(tbl: pl.DataFrame) -> float:
    """The fraction of parameters whose 90 percent interval contains the truth."""
    if tbl.is_empty():
        return 0.0
    return float(tbl["covered"].sum() / tbl.height)


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

COVERED, MISSED = "#2a78d6", "#d03b3b"      # categorical slot 1; status critical
TRUTH_INK = "#0b0b0b"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"


def dot_whisker(tbl: pl.DataFrame) -> go.Figure:
    """One row per parameter: the interval as a whisker, the truth as a mark on top of it.

    Ordered worst-first by how far the truth sits outside the interval, so the rows the
    model got wrong are the rows at the top of the page rather than somewhere in the middle
    of sixty.
    """
    miss = pl.max_horizontal(pl.col("lo90") - pl.col("truth"), pl.col("truth") - pl.col("hi90"))
    ordered = tbl.with_columns(miss.alias("_miss")).sort("_miss", descending=True)
    source = tbl["source"][0] if tbl.height else "prior"
    n_missed = tbl.height - int(tbl["covered"].sum())

    fig = go.Figure()
    for covered, color, label in ((False, MISSED, "truth outside the interval"),
                                  (True, COVERED, "truth inside the interval")):
        s = ordered.filter(pl.col("covered") == covered)
        if not s.height:
            continue
        fig.add_scatter(
            x=s["post_mean"].to_list(), y=s["parameter"].to_list(), mode="markers",
            name=label, legendgroup=label,
            marker={"size": 9, "color": color, "line": {"color": SURFACE, "width": 2}},
            error_x={"type": "data", "symmetric": False,
                     "array": (s["hi90"] - s["post_mean"]).to_list(),
                     "arrayminus": (s["post_mean"] - s["lo90"]).to_list(),
                     "color": color, "thickness": 2, "width": 0},
            customdata=np.stack([s["truth"], s["lo90"], s["hi90"]], axis=-1),
            hovertemplate=("<b>%{y}</b><br>truth %{customdata[0]:.2f}<br>"
                           "mean %{x:.2f}  ·  90%% [%{customdata[1]:.2f}, "
                           "%{customdata[2]:.2f}]<extra></extra>"),
        )
    fig.add_scatter(
        x=ordered["truth"].to_list(), y=ordered["parameter"].to_list(), mode="markers",
        name="generating truth", marker={"symbol": "line-ns-open", "size": 11,
                                         "color": TRUTH_INK, "line": {"width": 2}},
        hoverinfo="skip")
    kind = ("Prior coverage" if source == "prior" else "Parameter recovery")
    note = ("rung 0 has no fit, so these are prior intervals"
            if source == "prior" else "90% posterior intervals")
    # Sixty term names on the y-axis, the longest around 47 characters. Sizing the gutter
    # off the longest label is the difference between a readable chart and clipped names.
    label_px = max((len(p) for p in tbl["parameter"]), default=20) * 6.6
    height = max(420, 20 * tbl.height + 180)
    fig.update_layout(
        title={"text": f"{kind}: truth against the 90% interval",
               "subtitle": {"text": (f"{note} · {coverage(tbl):.0%} of {tbl.height} "
                                     f"parameters covered · {n_missed} outside"),
                            "font": {"color": INK_2, "size": 13}},
               "font": {"color": INK, "size": 18}, "x": 0.02, "xanchor": "left"},
        width=int(560 + label_px), height=height,
        margin={"l": int(label_px + 16), "r": 24, "t": 110, "b": 56},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": INK_2},
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1 + 52 / height,
                "font": {"color": INK_2}},
        xaxis={"title": {"text": "Log-odds per unit", "font": {"color": INK_2, "size": 12}},
               "gridcolor": GRID, "zerolinecolor": AXIS, "linecolor": AXIS,
               "tickfont": {"color": MUTED}},
        yaxis={"autorange": "reversed", "showgrid": False, "linecolor": AXIS,
               "tickfont": {"color": MUTED, "size": 11}},
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": INK}},
    )
    return fig


def write_outputs(tbl: pl.DataFrame, out_dir: Path = REPORT) -> tuple[Path, Path]:
    """report/recovery.csv (one row per parameter) and .html (Plotly, works offline)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, html = out_dir / "recovery.csv", out_dir / "recovery.html"
    tbl.select(list(RECOVERY_SCHEMA)).write_csv(csv)
    # include_plotlyjs=True inlines plotly.js: bigger file, but it opens with the wifi off.
    dot_whisker(tbl).write_html(html, include_plotlyjs=True, full_html=True)
    return csv, html


def run(*, n_draws: int = N_DRAWS, seed: int = SEED,
        posterior: Path | None = POSTERIOR) -> pl.DataFrame:
    """The one call the report assembler makes."""
    draws, source = coefficient_draws(n_draws=n_draws, seed=seed, posterior=posterior)
    return recover(truth_coefficients(), draws, source=source)


def main() -> int:
    tbl = run()
    csv, html = write_outputs(tbl)
    source = tbl["source"][0]
    missed = tbl.filter(~pl.col("covered"))
    print(f"{'prior' if source == 'prior' else 'posterior'} intervals · "
          f"{coverage(tbl):.1%} of {tbl.height} parameters covered "
          f"(SPEC §11 bar is {COVERAGE_BAR:.0%})")
    if source == "prior":
        print("  rung 0: no MCMC ran, so this is prior coverage, not parameter recovery")
    with pl.Config(tbl_rows=missed.height + 1):
        print(missed.select("parameter", "truth", "lo90", "hi90") if missed.height
              else "  every parameter covered")
    print(f"wrote {csv.relative_to(schema.ROOT)} and {html.relative_to(schema.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

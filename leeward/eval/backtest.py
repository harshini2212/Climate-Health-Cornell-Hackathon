"""`make backtest` -- the heat model against NYC's real daily heat ED visits, summers 2017-2021.

    python -m leeward.eval.backtest --prior     # rung 0, the prior means   (make backtest)
    python -m leeward.eval.backtest --fitted    # the make-fit posterior     (make backtest-fitted)

One of the two is required: which model is scored is never decided by whether a posterior
happens to exist in data/.

Everything else in `eval/` scores the model against the simulator, which was written by the
same people who wrote the priors. This is the first check against something nobody here
made: NYC Health's syndromic count of heat-illness ED visits, one number per day, citywide
(`leeward/ingest/sources/heat_syndrome.py`).

What is predicted. For each June-August day, the model's heat-need probability, averaged
over the cohort -- which is re-homed by ZIP in proportion to ACS veterans, so the average is
the ZIP-weighted one. The heat log-odds are exactly `design.py`'s heat column times the
coefficients: the 82 F hinge with lags 0-3, the hot-day interactions, and every person term,
fed the day's real LaGuardia heat index. The 10,000 x 460 veteran-day design matrix is never
built; with one citywide thermometer the heat column splits into

    eta[v, t] = person[v] + hot_day[t] * hot_increment[v] + sum_l delta_l * heat_x[t - l]

and the test suite checks that split against `design.build` over every veteran-day.

What is reported, to `report/backtest.json` and one offline chart, `report/backtest.html`:

* **Rank correlation** (Spearman) of predicted intensity with observed visits, per year, and
  against two baselines: the lag terms alone, and the raw temperature. A model that cannot
  beat the thermometer is not adding anything on this test.
* **Lag structure.** The cross-correlation of predicted(t) with visits(t + k), k = 0-3; the
  observed series' rank correlation with heat_x at each lag; and -- the real lag evidence --
  a Poisson distributed-lag fit of observed visits on heat_x lags 0-3, whose normalised shape
  is set beside the model's `delta_heat` shape. Days below the 82 F hinge with no heat in
  the three before all score the calm baseline and tie; `n_days_at_baseline` says how many.
* **Calibration in the large**: predicted heat-need rate per veteran-day over observed heat
  ED visits per resident-day. These are not the same event -- a heat *need* is "this person
  should get a call", an ED visit is the tail of it -- so the ratio is reported as what it
  is, not as a pass/fail.
* **Spatial**, when the EHDP table is present: the cohort's predicted hot-day risk per
  community district against EHDP's real heat ED visit rate there, 2018-22 -- the nearest
  neighbourhood-level window the portal publishes.

It refuses to run on anything but the real series. `require_real` rejects the simulator's
outcome table, any `_synthetic` column, and any frame whose `source` is not NYC Health's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

from leeward import schema
from leeward.ingest.hazards import HOT_DAY_F
from leeward.ingest.sources import ehdp_heat, heat_syndrome
from leeward.model import design, hazard, priors, score
from leeward.schema import NEEDS

REF = schema.DATA / "reference"
REPORT = schema.ROOT / "report"
OBSERVED = REF / f"{heat_syndrome.STEM}.parquet"
EHDP = REF / f"{ehdp_heat.STEM}.parquet"
GEO = REF / f"{ehdp_heat.XWALK_STEM}.parquet"

YEARS = tuple(range(2017, 2022))
SUMMER_MONTHS = (6, 7, 8)
LAGS = design.HEAT_LAGS
HEAT = NEEDS.index("heat")
#: The spatial check scores every neighbourhood on the same four-day 92 F heat wave, so the
#: ranking reflects who lives there, not which day was picked.
SPATIAL_WAVE_F = 92.0
#: EHDP 2443, heat ED visits by community district, 2018-22: the only neighbourhood-level
#: heat outcome that overlaps 2017-2021 (the UHF42 yearly series stop in 2014).
SPATIAL_INDICATOR, SPATIAL_PERIOD = 2443, "2018-22"
COHORT_SEED = 0


class SyntheticOutcomes(ValueError):
    """The back-test was handed something other than NYC Health's real series."""


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

def require_real(observed: pl.DataFrame) -> pl.DataFrame:
    """Return `observed` unchanged if it is the real heat-syndrome series; raise otherwise."""
    if "veteran_id" in observed.columns:
        raise SyntheticOutcomes(
            "observed outcomes have a veteran_id column: that is the simulator's outcome table "
            "(one row per synthetic veteran-day), not NYC Health's citywide series")
    flags = [c for c in observed.columns if c.endswith("_synthetic")]
    if flags:
        raise SyntheticOutcomes(f"observed outcomes carry synthetic flags {flags}")
    if "source" not in observed.columns:
        raise SyntheticOutcomes("observed outcomes have no `source` column, so nothing says "
                                f"they came from {heat_syndrome.REPO}")
    other = sorted(set(observed["source"].unique().to_list()) - {heat_syndrome.SOURCE})
    if other:
        raise SyntheticOutcomes(f"observed outcomes come from {other}, not "
                                f"{heat_syndrome.SOURCE!r}")
    files = sorted(set(observed["file"].unique().to_list()) - set(heat_syndrome.FILES)
                   ) if "file" in observed.columns else ["<no file column>"]
    if files:
        raise SyntheticOutcomes(f"observed rows come from {files}, not the published files "
                                f"{list(heat_syndrome.FILES)}")
    missing = {"date", "max_temp_f", "heat_ed_visits"} - set(observed.columns)
    if missing:
        raise ValueError(f"observed outcomes are missing {sorted(missing)}")
    return observed


def load_observed(path: Path = OBSERVED, manifest: Path = REF / "manifest.json") -> pl.DataFrame:
    """The committed series, refused unless its bytes still match the fetch in the manifest.
    The `source` tag is a string anyone can write; the hash is what the fetcher recorded."""
    entry = json.loads(manifest.read_text(encoding="utf-8"))["heat_syndrome"]
    got = hashlib.sha256(path.read_bytes()).hexdigest()[: len(entry["sha256"])]
    if path.name != entry["file"] or got != entry["sha256"]:
        raise SyntheticOutcomes(f"{path} does not match the fetched {entry['file']} "
                                f"(sha256 {got}, manifest {entry['sha256']}); re-run "
                                "`scripts/fetch_sources.py --only heat_syndrome`")
    return require_real(pl.read_parquet(path))


# --------------------------------------------------------------------------- #
# Coefficients
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class Coef:
    B: np.ndarray       # (P, K) point estimate
    rung: int
    detail: str


def coefficients(posterior: Path | None = hazard.POSTERIOR) -> Coef:
    """The posterior mean if a fit exists (terms the rung did not fit are zero, as in
    `score.coefficients`), the prior means otherwise."""
    if posterior is not None and Path(posterior).exists():
        c = score.coefficients(posterior=posterior)
        return Coef(c.B.mean(axis=0), c.rung, f"{c.detail}; posterior mean")
    return Coef(priors.mean_matrix(), 0, "rung 0 prior means")


# --------------------------------------------------------------------------- #
# The prediction
# --------------------------------------------------------------------------- #

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def citywide_hazards(cohort: pl.DataFrame, temps: pl.DataFrame) -> pl.DataFrame:
    """A hazards frame with the day's citywide heat index in every ZIP and nothing else."""
    days = temps.select("date", heat_index_max_f=pl.col("max_temp_f").cast(pl.Float64))
    zips = cohort.select("modzcta").unique()
    return zips.join(days, how="cross").with_columns(
        pm25=pl.lit(0.0),
        hot_day=pl.col("heat_index_max_f") >= HOT_DAY_F,
        smoke_alert=pl.lit(False),
        flood_warning=pl.lit(False),
        flash_flood_emergency=pl.lit(False),
        floodnet_trip=pl.lit(False),
        surge_ft=pl.lit(0.0),
        evac_zone_ordered=pl.lit(0, dtype=pl.Int8),
        outage_frac=pl.lit(0.0),
        mail_delivery_disrupted=pl.lit(False),
    )


def calm_sites(cohort: pl.DataFrame, temps: pl.DataFrame) -> pl.DataFrame:
    """Every VA site open on every day: the back-test isolates heat."""
    return (cohort.select("facility_id").unique()
            .join(temps.select("date"), how="cross")
            .with_columns(site_down=pl.lit(False), site_dependent_services=pl.lit(False)))


def person_terms(cohort: pl.DataFrame, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(person, hot_increment), each (N,): heat log-odds on a calm day, and what a hot day at
    exactly the hinge adds. Read off `design.build` on two such days, so every person term
    and hot-day interaction comes from the shared mapping, not from a copy of it."""
    days = pl.DataFrame({"date": [date(2000, 1, 1), date(2000, 1, 2)],
                         "max_temp_f": [HOT_DAY_F - 20, HOT_DAY_F]})
    d = design.build(cohort, citywide_hazards(cohort, days), calm_sites(cohort, days))
    X = d.X.reshape(cohort.height, 2, design.P)
    lag_cols = design.TERM_SLICE["delta_heat"]
    assert not X[:, :, lag_cols].any(), "the reference days must carry no heat_x"
    eta = X @ B[:, HEAT]
    return eta[:, 0], eta[:, 1] - eta[:, 0]


def predict(cohort: pl.DataFrame, temps: pl.DataFrame, B: np.ndarray) -> pl.DataFrame:
    """One row per day in `temps`: heat_x lags, hot_day, the lag-term log-odds, and the
    cohort-mean heat-need probability (`intensity`). `baseline` is that mean on a calm day."""
    temps = temps.select("date", pl.col("max_temp_f").cast(pl.Float64)).sort("date")
    one_zip = cohort.head(1).select("modzcta")
    lagged = design._hazard_frame(citywide_hazards(one_zip, temps)).sort("date")
    heat_x = lagged.select([f"heat_x{i}" for i in range(LAGS)]).to_numpy()
    hot = lagged["hot_day"].cast(pl.Float64).to_numpy()

    delta = B[design.TERM_SLICE["delta_heat"], HEAT]
    lag_logit = heat_x @ delta
    person, inc = person_terms(cohort, B)
    p = _sigmoid(person[:, None] + inc[:, None] * hot[None, :] + lag_logit[None, :])
    return temps.with_columns(
        hot_day=lagged["hot_day"],
        **{f"heat_x{i}": pl.Series(heat_x[:, i]) for i in range(LAGS)},
        lag_logit=pl.Series(lag_logit),
        intensity=pl.Series(p.mean(axis=0)),
        baseline=pl.lit(float(_sigmoid(person).mean())),
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman's rho with average ranks for ties. NaN under 3 points or if a side is constant."""
    if len(a) < 3:
        return float("nan")
    ra = pl.Series(a).rank("average").to_numpy()
    rb = pl.Series(b).rank("average").to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def poisson_fit(X: np.ndarray, y: np.ndarray, iters: int = 100) -> np.ndarray:
    """Poisson regression with a log link by IRLS. X includes the intercept column."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.mean(), 1e-9))
    for _ in range(iters):
        mu = np.exp(X @ beta)
        z = X @ beta + (y - mu) / mu
        new = np.linalg.solve(X.T @ (mu[:, None] * X), X.T @ (mu * z))
        done = np.max(np.abs(new - beta)) < 1e-10
        beta = new
        if done:
            break
    return beta


def _summer_days() -> list[date]:
    return [d for y in YEARS for d in _range(date(y, SUMMER_MONTHS[0], 1),
                                             date(y, SUMMER_MONTHS[-1] + 1, 1))]


def _range(start: date, stop: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((stop - start).days)]


def _needed_days() -> list[date]:
    """Summer, plus the lag days before it and the cross-correlation days after it."""
    out = []
    for y in YEARS:
        start = date(y, SUMMER_MONTHS[0], 1) - timedelta(days=LAGS - 1)
        stop = date(y, SUMMER_MONTHS[-1] + 1, 1) + timedelta(days=LAGS - 1)
        out += _range(start, stop)
    return out


def _shift(daily: pl.DataFrame, col: str, k: int) -> pl.DataFrame:
    """`col` from k days later, keyed on the earlier date."""
    return daily.select(pl.col("date") - pl.duration(days=k), pl.col(col).alias(f"{col}_lead"))


def _spatial(cohort: pl.DataFrame, B: np.ndarray, ehdp: pl.DataFrame, geo: pl.DataFrame,
             min_veterans: int) -> dict:
    other = sorted(set(ehdp["source"].unique().to_list()) - {ehdp_heat.SOURCE})
    if other:
        raise SyntheticOutcomes(f"neighbourhood rates come from {other}, not EHDP")
    person, inc = person_terms(cohort, B)
    wave = B[design.TERM_SLICE["delta_heat"], HEAT].sum() * min(
        (SPATIAL_WAVE_F - design.HEAT_HINGE_F) / design.HEAT_UNIT_F, design.HEAT_CAP)
    pred = (cohort.select("modzcta").with_columns(p=pl.Series(_sigmoid(person + inc + wave)))
            .join(geo.select("modzcta", "cd"), on="modzcta", how="inner")
            .group_by("cd").agg(predicted=pl.col("p").mean(), n_veterans=pl.len())
            .filter(pl.col("n_veterans") >= min_veterans))
    obs = (ehdp.filter((pl.col("indicator_id") == SPATIAL_INDICATOR) & (pl.col("geo_type") == "CD")
                       & (pl.col("measure") == "Estimated annual rate")
                       & (pl.col("time_period") == SPATIAL_PERIOD) & pl.col("value").is_not_null())
           .select(pl.col("geo_id").alias("cd"), "geo_name",
                   pl.col("value").alias("observed_rate_per_100k")))
    both = pred.join(obs, on="cd", how="inner").sort("cd")
    return {
        "what": (f"cohort-mean heat-need probability on a four-day {SPATIAL_WAVE_F:.0f} F wave, "
                 "per community district, against EHDP's heat ED visits per 100,000 residents "
                 f"per year (indicator {SPATIAL_INDICATOR}, {SPATIAL_PERIOD}; suppressed "
                 "districts dropped)"),
        "n_neighbourhoods": both.height,
        "min_veterans_per_neighbourhood": min_veterans,
        "spearman": spearman(both["predicted"].to_numpy(),
                             both["observed_rate_per_100k"].to_numpy()),
        "rows": both.to_dicts(),
    }


def backtest(observed: pl.DataFrame, cohort: pl.DataFrame, B: np.ndarray, *, rung: int,
             detail: str, population: float, ehdp: pl.DataFrame | None = None,
             geo: pl.DataFrame | None = None, min_veterans: int = 30,
             ) -> tuple[dict, pl.DataFrame]:
    """(report, one row per scored summer day). Raises on synthetic or incomplete input."""
    obs = require_real(observed).select("date", "max_temp_f", "heat_ed_visits").sort("date")
    have = set(obs["date"].to_list())
    holes = [d for d in _needed_days() if d not in have]
    if holes:
        raise ValueError(f"the observed series is missing {len(holes)} needed days, e.g. "
                         f"{', '.join(d.isoformat() for d in holes[:5])}")

    daily = predict(cohort, obs.select("date", "max_temp_f"), B).join(
        obs.select("date", "heat_ed_visits"), on="date")
    out = daily.filter(pl.col("date").is_in(_summer_days())).sort("date")
    y = out["heat_ed_visits"].to_numpy().astype(float)
    pred = out["intensity"].to_numpy()

    by_year = {str(yr): spearman(g["intensity"].to_numpy(), g["heat_ed_visits"].to_numpy())
               for (yr,), g in out.group_by(pl.col("date").dt.year(), maintain_order=True)}

    xcorr = []
    for k in range(LAGS):
        led = out.select("date", "intensity").join(_shift(daily, "heat_ed_visits", k), on="date")
        xcorr.append(spearman(led["intensity"].to_numpy(), led["heat_ed_visits_lead"].to_numpy()))
    heat_x = out.select([f"heat_x{i}" for i in range(LAGS)]).to_numpy()
    coef = poisson_fit(np.column_stack([np.ones(len(y)), heat_x]), y)
    delta = B[design.TERM_SLICE["delta_heat"], HEAT]

    def shape(v: np.ndarray) -> list[float] | None:
        """Normalised lag weights; None if any is negative, where a share means nothing."""
        return (v / v.sum()).tolist() if (v >= 0).all() and v.sum() > 0 else None

    baseline = float(out["baseline"][0])
    # Days with no heat today or in the three days before: the observed match for the model's
    # calm-day baseline, which has every heat_x lag at zero.
    calm = pl.all_horizontal([pl.col(f"heat_x{i}") == 0 for i in range(LAGS)])
    cool = out.filter(calm)["heat_ed_visits"].mean()
    pred_rate, obs_rate = float(pred.mean()), float(y.mean() / population)
    pred_excess = float((pred - baseline).mean())
    obs_excess = float((y.mean() - (cool or 0.0)) / population)

    report = {
        "what": ("NYC daily heat-syndrome ED visits against the model's predicted heat-need "
                 "intensity, June-August 2017-2021"),
        "observed": {"source": heat_syndrome.REPO, "files": list(heat_syndrome.FILES),
                     "total_heat_ed_visits": int(y.sum()),
                     "max_day": out.sort("heat_ed_visits", descending=True)
                                   .select(pl.col("date").cast(pl.String), "max_temp_f",
                                           "heat_ed_visits")
                                   .row(0, named=True)},
        "model": {"rung": rung, "detail": detail, "hinge_f": design.HEAT_HINGE_F,
                  "delta_heat": delta.tolist(), "n_veterans": cohort.height},
        "window": {"years": [YEARS[0], YEARS[-1]], "months": list(SUMMER_MONTHS)},
        "n_days": out.height,
        "n_days_at_baseline": out.filter(calm).height,
        "rank_correlation": {
            "spearman": spearman(pred, y),
            "spearman_lag_terms_only": spearman(out["lag_logit"].to_numpy(), y),
            "spearman_temperature_baseline": spearman(out["max_temp_f"].to_numpy(), y),
            "by_year": by_year,
        },
        "lag_structure": {
            "predicted_vs_observed_xcorr": xcorr,
            "observed_vs_heat_x": [spearman(heat_x[:, k], y) for k in range(LAGS)],
            "observed_poisson_intercept": float(coef[0]),
            "observed_poisson_coef": coef[1:].tolist(),
            "observed_shape": shape(coef[1:]),
            "model_shape": shape(delta),
        },
        "calibration_in_the_large": {
            "predicted_rate_per_person_day": pred_rate,
            "observed_rate_per_person_day": obs_rate,
            "ratio": pred_rate / obs_rate if obs_rate > 0 else None,
            "predicted_excess_per_person_day": pred_excess,
            "observed_excess_per_person_day": obs_excess,
            "excess_ratio": pred_excess / obs_excess if obs_excess > 0 else None,
            "population": population,
            "note": ("predicted = mean heat-need probability per veteran-day; observed = heat "
                     "ED visits per NYC resident-day. Different events and populations: a "
                     "need is a reason to call, an ED visit is its tail, and veterans are "
                     "older than the city. So this ratio is a scale check, not a calibration verdict. "
                     "Excess = above the calm-day baseline, which for both sides means no "
                     "heat_x on the day or the three before it."),
        },
        "spatial": (_spatial(cohort, B, ehdp, geo, min_veterans)
                    if ehdp is not None and geo is not None else None),
    }
    return report, out


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e8e7e1", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")


def figure(out: pl.DataFrame, report: dict) -> go.Figure:
    """Predicted intensity against observed visits, one colour per summer."""
    fig = go.Figure()
    for (yr,), g in out.group_by(pl.col("date").dt.year(), maintain_order=True):
        fig.add_trace(go.Scatter(
            x=g["intensity"] * 100, y=g["heat_ed_visits"], mode="markers", name=str(yr),
            marker={"size": 8, "color": SERIES[YEARS.index(yr)], "opacity": 0.8,
                    "line": {"width": 1, "color": "#ffffff"}},
            customdata=np.column_stack([g["date"].cast(pl.String), g["max_temp_f"]]),
            hovertemplate=("%{customdata[0]}<br>heat index %{customdata[1]} F<br>"
                           "predicted %{x:.3f}% per veteran-day<br>"
                           "%{y} heat ED visits<extra></extra>")))
    rc = report["rank_correlation"]
    fig.update_layout(
        title={"text": (f"Heat model vs NYC heat ED visits, Jun-Aug 2017-2021 "
                        f"(rung {report['model']['rung']}): Spearman {rc['spearman']:.2f}, "
                        f"temperature alone {rc['spearman_temperature_baseline']:.2f}"),
               "font": {"color": INK, "size": 15}},
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", height=520,
        legend={"orientation": "h", "x": 0, "y": -0.15, "font": {"color": INK_2}},
        xaxis={"title": {"text": "Predicted heat-need probability, % per veteran-day",
                         "font": {"color": INK_2}},
               "gridcolor": GRID, "linecolor": AXIS, "tickfont": {"color": MUTED}},
        yaxis={"title": {"text": "Heat-syndrome ED visits, citywide", "font": {"color": INK_2}},
               "gridcolor": GRID, "linecolor": AXIS, "zerolinecolor": AXIS,
               "tickfont": {"color": MUTED}},
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": INK}},
    )
    return fig


def write(report: dict, out: pl.DataFrame, out_dir: Path = REPORT) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    js, html = out_dir / "backtest.json", out_dir / "backtest.html"
    js.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    # include_plotlyjs=True inlines plotly.js: bigger file, but it opens with the wifi off.
    figure(out, report).write_html(html, include_plotlyjs=True, full_html=True)
    return js, html


def _fmt_shape(shape: list[float] | None) -> str:
    return " ".join(f"{v:.2f}" for v in shape) if shape else "n/a (a negative lag weight)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the heat model against real NYC heat ED visits")
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--prior", action="store_true", help="rung 0: the prior means")
    which.add_argument("--fitted", action="store_true", help="the posterior from `make fit`")
    ap.add_argument("--posterior", type=Path, default=hazard.POSTERIOR)
    ap.add_argument("--out", type=Path, default=REPORT)
    args = ap.parse_args(argv)
    if args.fitted and not args.posterior.exists():
        ap.error(f"--fitted needs {args.posterior}; run `make fit`, or score the priors "
                 "with --prior")

    from leeward.cohort import build as cohort_build

    coef = coefficients(None if args.prior else args.posterior)
    cohort = cohort_build.build(seed=COHORT_SEED)
    population = float(pl.read_parquet(REF / "nyc_modzcta.parquet")["pop_est"].sum())
    ehdp = pl.read_parquet(EHDP) if EHDP.exists() else None
    geo = pl.read_parquet(GEO) if GEO.exists() else None
    report, out = backtest(load_observed(), cohort, coef.B, rung=coef.rung, detail=coef.detail,
                           population=population, ehdp=ehdp, geo=geo)
    js, html = write(report, out, args.out)

    rc, lag = report["rank_correlation"], report["lag_structure"]
    citl = report["calibration_in_the_large"]
    print(f"rung {coef.rung}: {coef.detail}; {report['n_days']} summer days, "
          f"{report['n_days_at_baseline']} of them at the calm baseline")
    print(f"  Spearman {rc['spearman']:.3f}  (lag terms only {rc['spearman_lag_terms_only']:.3f}, "
          f"temperature alone {rc['spearman_temperature_baseline']:.3f})")
    print("  xcorr lags 0-3 " + " ".join(f"{v:.3f}" for v in lag["predicted_vs_observed_xcorr"]))
    print(f"  observed lag shape {_fmt_shape(lag['observed_shape'])}  "
          f"vs model {_fmt_shape(lag['model_shape'])}")
    nan = float("nan")
    print(f"  calibration in the large: predicted need per veteran-day / observed ED visits "
          f"per resident-day = {citl['ratio'] or nan:,.0f}x (excess {citl['excess_ratio'] or nan:,.0f}x)"
          f"; different events, a scale check")
    if report["spatial"]:
        sp = report["spatial"]
        print(f"  spatial Spearman {sp['spearman']:.3f} over {sp['n_neighbourhoods']} community districts")
    print(f"  wrote {js} and {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

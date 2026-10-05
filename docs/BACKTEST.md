# Heat back-test

The heat model against outcomes nobody on this project made: NYC Health's daily
heat-syndrome ED visits, citywide, June–August 2017–2021. 460 days, 2,617 visits. Method,
caveats and code are in `leeward/eval/backtest.py`; the series is in
`data/reference/nyc_heat_ed_daily.parquet` (see `docs/sources.md`).

```bash
make backtest          # rung 0, the prior means
make backtest-fitted   # the `make fit` posterior; fails if there is none
```

Both write `report/backtest.json` and `report/backtest.html`. Rows are append-only: record a
new one rather than editing an old one.

## Results

Run 2026-10-05 at `9f51e24` (the model, cohort, simulator and back-test code are unchanged
at `370faef`). Cohort: 10,000 synthetic veterans, seed 0. Each run was repeated and
reproduced exactly.

| | Rung 0 (prior means) | Rung 1 (posterior mean) |
| --- | --- | --- |
| Fit | — | `make fit`, seed 0: r-hat max 1.005, 0 divergences, ESS ≥ 2,556; 19 terms fitted, 16 interactions left out of the prediction |
| **Spearman, predicted vs observed** | **0.723** | **0.736** |
| Spearman, lag terms only | 0.713 | 0.736 |
| Spearman, LaGuardia temperature alone | 0.729 | 0.729 |
| Cross-correlation, predicted(t) vs visits(t+k), k = 0–3 | 0.72 / 0.53 / 0.32 / 0.18 | 0.74 / 0.54 / 0.34 / 0.20 |
| `delta_heat` (log-odds per 10 °F over 82 °F), lags 0–3 | 0.50 / 0.30 / 0.20 / 0.10 | 1.31 / 0.40 / 0.33 / 0.24 |
| Its normalised shape | 0.45 / 0.27 / 0.18 / 0.09 | 0.57 / 0.17 / 0.15 / 0.11 |
| **Observed shape** (Poisson distributed-lag fit, same for both) | 0.72 / 0.22 / 0.01 / 0.05 | 0.72 / 0.22 / 0.01 / 0.05 |
| Mean predicted heat need, per veteran-day | 3.75% | 12.04% |
| Predicted need ÷ observed ED visits, per person-day | 55,606× | 178,688× |
| Same, above the calm-day baseline | 47,923× | 148,460× |
| Spatial: hot-day risk vs EHDP heat ED rate, 2018-22 | 0.097 (56 districts) | 0.147 (56 districts) |

## What it says

- **Ranking hot days is the thermometer's job, and the model matches it.** Rung 1 edges
  temperature alone, 0.736 against 0.729, and rung 0 sits just under it. With one citywide
  thermometer this test can only see the 82 °F hinge and the lag curve. At rung 1 the lag
  terms alone carry the whole score (0.736 either way). The cohort's person terms do not
  change from day to day, and the hot-day interactions are not in rung 1's prediction.
- **Fitting moved the lag curve toward reality, but not far enough.** Real heat ED visits
  are front-loaded: 72% of the weight is same-day, and almost none falls on days 2–3. The
  fitted curve put more weight on day 0 (57%, from 45%). It still gives days 2–3 a quarter of
  the weight, where the real series gives them 6%.
- **The absolute level is not anchored to anything real, and fitting made it worse.** The
  ratio compares different events: a heat *need*, meaning "this veteran should get a call",
  against an ED visit, which is that need's tail. So it is a scale check, not a calibration
  verdict. Even so, rung 1 says one veteran in eight has a heat need on an average NYC
  summer day. It was fitted to a simulated panel in which 5.84% of all veteran-days carry a
  heat need, so its level is the simulator's. A real anchor for the heat need's base rate is
  still missing.
- **The cohort barely ranks neighbourhoods the way real heat ED visits do.** Spearman is
  0.10 at rung 0 and 0.15 at rung 1, across 56 community districts. This is the only place
  the back-test sees who lives where. The re-homed cohort's heat-risk mix does not track
  where heat illness actually reaches the ED.

## Caveats

- **Syndromic counts.** These are suspected heat illness, not confirmed. Residents treated
  outside the city are missed. 2021 is the publisher's "live" file.
- **Calm-baseline ties.** 46 of the 460 days have no heat that day or in the three before
  it, so they score the calm baseline and tie. 118 days are below 82 °F on the day itself.
- **Rung 1 learned from the simulator, not from these data.** Its posterior is
  `data/posterior.nc`, which is not committed. `make cohort && make fit` rebuilds it with
  seed 0; this run took 65 minutes on a laptop.
- **Spatial data.** It uses EHDP indicator 2443 (2018-22) by community district, the
  nearest neighbourhood-level window to 2017–2021. Each MODZCTA goes wholly to its
  largest-overlap district.

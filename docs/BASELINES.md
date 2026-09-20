# Baselines

One row per recorded run of the whole pipeline. A number is only useful if you can say what
it was before, so this is the before.

```bash
python scripts/baseline.py --label "rung 1"   # run the pipeline, record a row
python scripts/baseline.py --compare          # diff the last two, change nothing
```

Each row names the git sha, the model rung and the seed's scenario, so a number that moves
can be attributed to a change rather than to the weather. Rows are append-only; do not edit
an old one, record a new one.

**Determinism is verified, not assumed.** Two independent full runs at `36b55ce` produced
byte-identical metrics — every ECE, the coverage, harm averted, the fairness ratio, the tier
mix. So a number that moves between rows moved because the code changed, not because the run
did. Re-check this whenever a stage starts drawing randomness.

**What "better" means, per metric:** ECE lower, coverage higher, harm averted higher, lift
higher, fairness flagged lower, find-out share higher (it is zero when the model has no
opinion about uncertainty), total seconds lower. Everything else is context, not a score.

---

## rung 0 · prior-only · Round A + B — `36b55ce`

*2026-09-20T03:49:59Z · Python 3.11.14 · arm64*

| | |
| --- | --- |
| **Harm averted, 40 calls/day** | **28.76** vs random 3.74, rank_by_age 3.37, rank_by_chronic 6.79 — **4.24× the best baseline** |
| **Calibration (ECE, bar 0.03)** | max **0.0075** · mean 0.0049 |
| **Parameter coverage (bar 0.90)** | **0.9833** |
| **Fairness** | 0 flagged of 29 groups · worst FNR ratio 1.035 |
| **Model fit** | prior-only, so no r-hat and no divergences |

**ECE by need** — access_loss 0.0067 · breathing 0.0011 · heat 0.0075 · mental 0.0044 · treatment_gap 0.005

**Simulated base rate per day** — access_loss 0.436% · breathing 0.669% · heat 5.837% · mental 0.537% · treatment_gap 1.809%

**Panel** — 10,000 veterans across 175 ZIPs · 6,000,000 scored rows · 312,281 actions

**Medication** — 4.37 drugs each · 72.0% heat-impairing · 18.4% on the CDC pair · 10.8% cold-chain · 5.5% controlled

**Tier mix** — act-now 5.97% · find-out 0.144%

**Pipeline** — hazards 0.16s · cohort 0.3s · simulate 1.35s · score 9.04s · allocate 4.78s · report 6.1s · total 21.7s

---

The rung-0 baseline — your "before"

┌────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
│                            │                                                                                        │
├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ Harm averted, 40 calls/day │ 28.76 vs 6.79 rank-by-chronic, 3.74 random, 3.37 rank-by-age — 4.24× the best baseline │
├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ Calibration (ECE)          │ max 0.0075, mean 0.0049 — bar is 0.03                                                  │
├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ Parameter coverage         │ 0.9833 — bar is 0.90                                                                   │
├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ Fairness                   │ 0 flagged of 29 groups, worst FNR ratio 1.035                                          │
├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ Find-out share             │ 0.144% of actions                                                                      │
├────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ Whole pipeline             │ 21.7 s                                                                                 │
└────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘

---

## Round C · do-by scheduling, race/ethnicity, dialysis 60, report screen — rung 0 — `42b37bf` *(uncommitted changes)*

*2026-09-20T05:13:43Z · Python 3.11.14 · arm64*

| | |
| --- | --- |
| **Harm averted, 40 calls/day** | **16.31** vs random 3.74, rank_by_age 3.37, rank_by_chronic 6.79 — **2.4× the best baseline** |
| **Calibration (ECE, bar 0.03)** | max **0.0075** · mean 0.005 |
| **Parameter coverage (bar 0.90)** | **0.9833** |
| **Fairness** | 0 flagged of 33 groups · worst FNR ratio 1.035 |
| **Model fit** | prior-only, so no r-hat and no divergences |

**ECE by need** — access_loss 0.0067 · breathing 0.0011 · heat 0.0075 · mental 0.0044 · treatment_gap 0.0055

**Simulated base rate per day** — access_loss 0.437% · breathing 0.669% · heat 5.837% · mental 0.537% · treatment_gap 1.859%

**Panel** — 10,000 veterans across 175 ZIPs · 6,000,000 scored rows · 326,679 actions

**Medication** — 4.37 drugs each · 72.0% heat-impairing · 18.4% on the CDC pair · 10.8% cold-chain · 5.5% controlled

**Tier mix** — act-now 7.31% · find-out 0.226%

**Pipeline** — hazards 0.22s · cohort 0.37s · simulate 2.02s · score 12.98s · allocate 8.01s · report 12.0s · total 35.6s

---

---

## Round C · do-by scheduling, race + ethnicity, dialysis 60, report screen — rung 0 — `42b37bf` *(uncommitted changes)*

*2026-09-20T05:17:39Z · Python 3.11.14 · arm64*

| | |
| --- | --- |
| **Harm averted, 40 calls/day** | **16.31** vs random 3.74, rank_by_age 3.37, rank_by_chronic 6.79 — **2.4× the best baseline** |
| **Per call actually made** | **0.524** — **3.09×**, spending 934 of 1200 available calls |
| **Calibration (ECE, bar 0.03)** | max **0.0075** · mean 0.005 |
| **Parameter coverage (bar 0.90)** | **0.9833** |
| **Fairness** | 0 flagged of 33 groups · worst FNR ratio 1.035 |
| **Model fit** | prior-only, so no r-hat and no divergences |

**ECE by need** — access_loss 0.0067 · breathing 0.0011 · heat 0.0075 · mental 0.0044 · treatment_gap 0.0055

**Simulated base rate per day** — access_loss 0.437% · breathing 0.669% · heat 5.837% · mental 0.537% · treatment_gap 1.859%

**Panel** — 10,000 veterans across 175 ZIPs · 6,000,000 scored rows · 326,679 actions

**Medication** — 4.37 drugs each · 72.0% heat-impairing · 18.4% on the CDC pair · 10.8% cold-chain · 5.5% controlled

**Tier mix** — act-now 7.31% · find-out 0.226%

**Pipeline** — not re-run · total Nones

---

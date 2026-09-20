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

**What "better" means, per metric:** within-day AUC higher and needs-above-0.65 higher (this
is the evidence); harm per call higher; find-out share higher (it is near zero when the model
has no opinion about uncertainty); coverage higher; ECE lower *but only read beside its
constant-predictor control*; fairness flagged lower *but only read beside reach and coverage*;
total seconds lower. Everything else is context, not a score.

**Two metrics are not evidence on their own, and the doc says so where they appear:** a
constant at the base rate beats the model on ECE, and the fairness flag has no reachable
failing state at this cohort's pooled FNR.

---

## Where we actually got to

Six recorded runs, `36b55ce` → `d6ef34e`, all rung 0. Read this before quoting anything: the
headline number went **down**, and it went down for a good reason. Two things improved
materially, one is flat, and one turned out never to have been evidence.

| | first | last | |
| --- | --- | --- | --- |
| **Find-out tier, share of actions** | 0.144% | **9.147%** | **64×** — the one large, unambiguous win |
| **Harm averted per call made** | not measured | **0.5185** | **3.05×** the best baseline |
| **Discrimination, within-day AUC** | not measured | heat **0.727**, treatment gap **0.715** | new evidence; 2 of 5 needs clear 0.65 |
| Harm averted per *day*, K=40 | 28.76 | 16.14 | **down 44%** — see below |
| Lift over the best baseline, per day | 4.24× | 2.38× | down, same reason |
| Calibration, worst ECE | 0.0075 | 0.0076 | flat — and not evidence, see below |
| Fairness groups audited | 29 | 33 | 0 flagged throughout, which is *also* not a pass |
| Whole pipeline | 21.7 s | 35.8 s | slower, and doing considerably more |

### The headline went down because the old one was inflated

`28.76` was measured when the allocator spent all forty calls every day, including days when
nothing was worth calling about. Do-by scheduling changed that: an action is offered on the
day it must be *done*, and on a calm day there is often nothing to do. **Seven of thirty
held-out days now avert zero on purpose**, and some days select two actions out of forty
available.

Per *day*, that reads as a 44% drop. Per *call actually made* — the honest denominator when a
system is allowed to do less — it is flat and slightly better than the baselines it always
beat: **489.4 harm averted on 934 calls, against 203.7 on all 1,200.** Same work, 22% fewer
calls.

So the claim to make is **"2.4× the harm on 22% fewer calls, 3.05× per call"**, not "4.2×".
The bigger number was real arithmetic on a worse policy.

### Two numbers stopped being evidence, and we found that ourselves

**Calibration.** A constant predictor at the base rate scores **ECE 0.0000 on every need** —
better than the model's 0.0011–0.0076. Rare events make everyone look calibrated. ECE was
never evidence that the model separates anybody, and `constant_ece` now ships beside it
everywhere, enforced by a guardrail.

**The fairness flag.** At a pooled FNR near 1, no group *can* exceed a 20% relative bar —
clearing it would take an FNR above 1. "0 of 33 flagged" was a metric with no failing state
being reported as a pass. The audit now carries reach, coverage and direction, and says so in
the note. It immediately surfaced something the flag never would: **Staten Island is reached
at half the cohort rate with 13% of the coverage.**

### What the model actually does, need by need

| Need | Within-day AUC | Lift @ 1% | Scaled Brier | |
| --- | --- | --- | --- | --- |
| **treatment gap** | 0.715 | **15.3×** | +0.042 | real |
| **heat** | 0.727 | **10.5×** | +0.108 | real |
| access loss | 0.645 | 4.1× | **−0.014** | worse than the base rate |
| mental | 0.605 | 2.8× | **−0.004** | worse than the base rate |
| breathing | 0.581 | 3.3× | +0.001 | near chance |

Lead with heat and treatment gap. Two of five needs are not yet worth claiming, and the
negative scaled Brier says so plainly.

### So: did we improve?

Yes, in two places, and both matter. The find-out tier went from firing on 0.14% of actions
to 9.15%, which is the difference between "uncertainty routes the unknown veteran to a cheap
call" being a slide and being a demonstrable behaviour — it took building the missingness
step the spec had always specified and nobody had written. And the evaluation got
substantially more honest: discrimination now exists, the calibration control ships beside
the curve, and the fairness audit reports a number that can move.

What did not improve is decision quality per call: **3.09× → 3.05×**, flat inside noise,
across everything from Round C onward. Do-by scheduling, race and ethnicity, the dialysis
rate and conditional imputation all changed *what the system knows and when it acts*, and
none of them moved how much harm a call averts. That is worth saying out loud rather than
letting a reader infer it from a table.

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

## Round C · do-by scheduling, race + ethnicity, dialysis 60, report screen — rung 0 — `42b37bf`

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

## before missingness — rung 0 — `fbe3a42`

*2026-09-20T05:25:56Z · Python 3.11.14 · arm64*

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

**Pipeline** — hazards 2.08s · cohort 0.49s · simulate 3.39s · score 17.26s · allocate 9.88s · report 21.55s · total 54.7s

---

## missingness + gap-aware find-out — rung 0 — `dcd1e3c`

*2026-09-20T06:08:03Z · Python 3.11.14 · arm64*

| | |
| --- | --- |
| **Harm averted, 40 calls/day** | **16.31** vs random 3.74, rank_by_age 3.37, rank_by_chronic 6.79 — **2.4× the best baseline** |
| **Per call actually made** | **0.5239** — **3.09×**, spending 934 of 1200 available calls |
| **Calibration (ECE, bar 0.03)** | max **0.0076** · mean 0.0051 |
| **Parameter coverage (bar 0.90)** | **0.9833** |
| **Fairness** | 0 flagged of 33 groups · worst FNR ratio 1.048 |
| **Model fit** | prior-only, so no r-hat and no divergences |

**ECE by need** — access_loss 0.0067 · breathing 0.0011 · heat 0.0076 · mental 0.0044 · treatment_gap 0.0055

**Simulated base rate per day** — access_loss 0.437% · breathing 0.669% · heat 5.837% · mental 0.537% · treatment_gap 1.859%

**Panel** — 10,000 veterans across 175 ZIPs · 6,000,000 scored rows · 326,966 actions

**Medication** — 4.37 drugs each · 72.0% heat-impairing · 18.4% on the CDC pair · 10.8% cold-chain · 5.5% controlled

**Tier mix** — act-now 7.05% · find-out 9.197%

**Pipeline** — hazards 0.36s · cohort 0.79s · simulate 6.46s · score 49.63s · allocate 16.73s · report 37.09s · total 111.1s

---

## conditional imputation of the gap — rung 0 — `07725ef` *(uncommitted changes)*

*2026-09-20T06:29:47Z · Python 3.11.14 · arm64*

| | |
| --- | --- |
| **Harm averted, 40 calls/day** | **16.14** vs random 3.74, rank_by_age 3.37, rank_by_chronic 6.79 — **2.38× the best baseline** |
| **Per call actually made** | **0.5185** — **3.05×**, spending 934 of 1200 available calls |
| **Calibration (ECE, bar 0.03)** | max **0.0076** · mean 0.005 |
| **Parameter coverage (bar 0.90)** | **0.9833** |
| **Fairness** | 0 flagged of 33 groups · worst FNR ratio 1.048 |
| **Model fit** | prior-only, so no r-hat and no divergences |

**ECE by need** — access_loss 0.0067 · breathing 0.0011 · heat 0.0076 · mental 0.0044 · treatment_gap 0.0055

**Simulated base rate per day** — access_loss 0.437% · breathing 0.669% · heat 5.837% · mental 0.537% · treatment_gap 1.859%

**Panel** — 10,000 veterans across 175 ZIPs · 6,000,000 scored rows · 327,397 actions

**Medication** — 4.37 drugs each · 72.0% heat-impairing · 18.4% on the CDC pair · 10.8% cold-chain · 5.5% controlled

**Tier mix** — act-now 7.08% · find-out 9.147%

**Pipeline** — hazards 0.26s · cohort 0.53s · simulate 3.82s · score 36.1s · allocate 10.37s · report 23.07s · total 74.2s

---

## Round D · missingness, discrimination, fairness ceiling — rung 0 — `c1679ef`

*2026-09-20T11:31:41Z · Python 3.11.14 · arm64*

| | |
| --- | --- |
| **Harm averted, 40 calls/day** | **16.14** vs random 3.74, rank_by_age 3.37, rank_by_chronic 6.79 — **2.38× the best baseline** |
| **Per call actually made** | **0.5185** — **3.05×**, spending 934 of 1200 available calls |
| **Calibration (ECE, bar 0.03)** | max **0.0076** · mean 0.005 |
| **Parameter coverage (bar 0.90)** | **0.9833** |
| **Fairness** | 0 flagged of 33 groups · worst FNR ratio 1.048 |
| **Model fit** | prior-only, so no r-hat and no divergences |

**ECE by need** — access_loss 0.0067 · breathing 0.0011 · heat 0.0076 · mental 0.0044 · treatment_gap 0.0055

**Simulated base rate per day** — access_loss 0.437% · breathing 0.669% · heat 5.837% · mental 0.537% · treatment_gap 1.859%

**Panel** — 10,000 veterans across 175 ZIPs · 6,000,000 scored rows · 327,397 actions

**Medication** — 4.37 drugs each · 72.0% heat-impairing · 18.4% on the CDC pair · 10.8% cold-chain · 5.5% controlled

**Tier mix** — act-now 7.08% · find-out 9.147%

**Pipeline** — hazards 0.2s · cohort 0.37s · simulate 2.05s · score 20.25s · allocate 10.08s · report 19.8s · total 52.8s

---

## Round D · re-recorded with discrimination and the ECE control — rung 0 — `d6ef34e` *(uncommitted changes)*

*2026-09-20T11:55:04Z · Python 3.11.14 · arm64*

| | |
| --- | --- |
| **Harm averted, 40 calls/day** | **16.14** vs random 3.74, rank_by_age 3.37, rank_by_chronic 6.79 — **2.38× the best baseline** |
| **Per call actually made** | **0.5185** — **3.05×**, spending 934 of 1200 available calls |
| **Discrimination (within-day AUC)** | heat **0.727** · treatment gap **0.715** · worst need 0.581 · 2 of 5 needs above 0.65 |
| **Calibration (ECE, bar 0.03)** | max **0.0076** · mean 0.005 — but a constant at the base rate scores 0.0, so this is not evidence on its own |
| **Parameter coverage (bar 0.90)** | **0.9833** |
| **Fairness** | 0 flagged of 33 groups · worst FNR ratio 1.048 |
| **Model fit** | prior-only, so no r-hat and no divergences |

**ECE by need** — access_loss 0.0067 · breathing 0.0011 · heat 0.0076 · mental 0.0044 · treatment_gap 0.0055

**Simulated base rate per day** — access_loss 0.437% · breathing 0.669% · heat 5.837% · mental 0.537% · treatment_gap 1.859%

**Panel** — 10,000 veterans across 175 ZIPs · 6,000,000 scored rows · 327,397 actions

**Medication** — 4.37 drugs each · 72.0% heat-impairing · 18.4% on the CDC pair · 10.8% cold-chain · 5.5% controlled

**Tier mix** — act-now 7.08% · find-out 9.147%

**Pipeline** — hazards 0.16s · cohort 0.29s · simulate 1.38s · score 16.92s · allocate 7.37s · report 9.68s · total 35.8s

---

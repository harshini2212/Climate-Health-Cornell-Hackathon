# eval


## 16:57 — eval/decision_quality.py
Harm averted at K∈{20,40,80} calls/day, Leeward vs age / n_chronic / random(seed 0), same K-call budget for all → report/decision_quality.{csv,html}; baselines + scoring tested, Leeward-beats-random skips only on the missing `leeward.decision.allocate` and needs `decision.severity.SEVERITY` + `decision.tau.TAU` from the api lane. make check green.

## 19:0x — tests no longer read data/
`tests/tables.py` builds every contract frame in memory from the `scripts/make_fixtures.py`
generators (seeded, geography still real from `data/reference/`); the five copies of the
`data/*.parquet` reader are gone, and a new guardrail fails any test file that adds a sixth.
Suite is now identical with fixtures, with the real 10k cohort, and with `data/` empty.
Fixed two assertions that were wrong for real data: a day is in the action list exactly when
someone is above the Everyday floor (the real Sandy run has 14 calm days that correctly
produce nothing, now pinned by two explicit tests), and the borough floor derives
floor(0.2 x 20) = 4 instead of hard-coding it.

## 20:2x — eval/{calibration,recovery,fairness,report}.py
Calibration on equal-mass bins per need (equal-width would park 99% of rung-0 rows in one
bin and report a flattering ECE); recovery against `truth.json` through `design.coef_matrix`
with a rung-1 `posterior.nc` reader tested by round-trip before rung 1 exists; fairness over
8 strata x 33 groups, FNR defined by the shipped `tiers` rule (act_now/find_out = reached),
every group binned on the whole window's edges so ECE gaps are comparable, flagged at
ratio > 1.20. `make report` assembles all of it plus decision_quality into `report/report.json`
and six offline charts in 2.8 s; `GET /report` validates. The fairness guardrail is live now.
Fixture run: ECE 0.0035-0.0080 per need (bar 0.03), prior coverage 98.3% of 60 parameters
(bar 90%), no group over the 20% gap. Rung 0 has no fit, so recovery is **prior coverage**,
not parameter recovery — carried by `model_rung: 0` + null `rhat_max`, and said in the CSV,
the chart subtitle and the console. The one miss is `psi_sitedown (treatment_gap)`: truth 1.5
against a prior 90% of [-0.30, 1.34], i.e. the prior understates the SiteDown term the demo
is built on. Worth a rung-1 fit, and worth saying out loud if asked.
Found and logged as **docs/PROMPTS.md 16** (not fixed — out of lane): `allocate()` spends
40 of 80 call slots on `check_in_call`, which `harm_averted` scores at 0 by construction, so
the Impact chart shows Leeward below random on the fixtures (49.6 vs 54.4) even though its
own picks scored as calls reach 76.6. The ranking is right; the metric and the budget
disagree about what a check-in is worth.

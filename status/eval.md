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

## 02:1x — eval/ablate.py
Five blocks zeroed at the coefficient (psi_sitedown; theta_sitedown_x_sitedependent; both
plus theta_sitedown_x_controlled; the medication block, derived from the design rather than
listed so a C3/C4 term joins it; delta_heat lags 1–3 with the same-day term kept), each
re-scored and re-run through the same `calibration.ece_overall` and the same `allocate` the
Impact chart uses. 6 models × 10k × 30 days in **30.5 s**, seeded. `make ablate` caches
`report/ablations.json`; `make report` reads it and says "not run" when it is absent, so the
Model report card fills without putting 30 s inside a target run every few edits.
**The headline is not what the deck assumes.** The SiteDown block as a whole is worth 6.97
severity-weighted events a day at 40 calls — but `psi_sitedown` *on its own* costs 3.24:
a flat lift for every patient of a closed station displaces veterans who actually have
events, and `theta_sitedown_x_sitedependent` is the part carrying the signal. Non-additive,
and worth saying out loud rather than rounding off. ECE barely moves (0.00503 → 0.00548 at
worst); at rung 0 with rare events it is not the discriminating number, harm averted is.
Gate: `ruff` clean, whole suite green **except** `test_the_slider_answers_inside_300ms…`,
which is the known load-bound gate — A/B'd at `origin/main` dc1c730 with none of my code in
it: 566 ms median there against 523 ms on this branch, with the model lane running 4-chain
NUTS at 577% CPU (load average 55). Not this diff. Re-run it on a quiet machine before
merging anything that touches the allocator.
Out of lane, not done: `ui/public/fixtures/report.json` still carries `ablations: []`, so
the offline UI fixture shows the empty card until the ui lane regenerates it after a
`make ablate && make report`.

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

## 02:2x — eval/ablate.py
Six blocks zeroed at the coefficient (every climate term, i.e. every `kind == "hazard"` one;
psi_sitedown; theta_sitedown_x_sitedependent; both of those plus theta_sitedown_x_controlled;
the medication block, derived from the design rather than listed so a C3/C4 term joins it;
delta_heat lags 1–3 with the same-day term kept), each re-scored and re-run through the same
`calibration.ece_overall` and the same `allocate` the Impact chart uses. 7 models × 10k ×
30 days in **~35 s quiet** (58 s while the model lane was running NUTS), seeded. `make ablate`
caches `report/ablations.json`; `make report` reads it and says "not run" when it is absent,
so the Model report card fills without putting half a minute inside a target run every few
edits. Harm averted per day at 40 calls, held-out days 2026-08-30 .. 09-28, full model 16.31:

| dropped | harm@40 | Δ |
| --- | --- | --- |
| every climate term | 0.77 | **−15.55** |
| the whole SiteDown block | 9.34 | **−6.97** |
| theta_sitedown_x_sitedependent | 16.59 | +0.27 |
| heat lags 1–3 | 16.67 | +0.35 |
| the medication block | 16.84 | +0.53 |
| psi_sitedown | 19.55 | **+3.24** |

The premise holds: without the weather the care team averts essentially nothing (0.77/day),
so the climate model is worth 15.55 severity-weighted events a day, not a rounding error.
**But the SiteDown result is not what the deck assumes.** The block as a whole is worth 6.97
— and `psi_sitedown` *on its own* costs 3.24. A flat lift for every patient of a closed
station displaces veterans who actually have events; `theta_sitedown_x_sitedependent` is the
part carrying the signal, and the two are strongly non-additive. Worth saying out loud rather
than rounding off, and worth re-running after a rung-1 fit — `psi_sitedown[treatment_gap]`
is the one parameter the 20:2x prior-coverage run missed (truth 1.5 against a prior 90% of
[−0.30, 1.34]), so the prior may simply have it too small to rank with. ECE barely moves
(0.00503 → 0.00599 at worst); at rung 0 with rare events it is not the discriminating
number, harm averted is.
Gate: `ruff` clean, whole suite green **except** `test_the_slider_answers_inside_300ms…`,
which is the known load-bound gate — A/B'd at `origin/main` dc1c730 with none of my code in
it: 566 ms median there against 523 ms on this branch, with the model lane running 4-chain
NUTS at 577% CPU (load average 55). Not this diff. Re-run it on a quiet machine before
merging anything that touches the allocator.
Out of lane, not done: `ui/public/fixtures/report.json` still carries `ablations: []`, so
the offline UI fixture shows the empty card until the ui lane regenerates it after a
`make ablate && make report`.

## 20:0x — eval/discrimination.py + the constant-predictor control
TRIPOD+AI's three legs are discrimination, calibration and clinical utility; the repo had the
last two. `leeward/eval/discrimination.py` adds the first: per need on the held-out window,
**within-day AUC** — the unweighted mean of per-day c-statistics — pooled AUC beside it,
PR-AUC, lift at the top 1%/10% against its ceiling, and the Brier skill score. Mann-Whitney
rank identity with tie-averaged ranks, no scikit-learn. Within-day is the headline because the
list is chosen within a day; pooled is partly "was today a heat wave". The estimand has a name
and a citation — within-cluster concordance, van Klaveren et al. BMC Med Res Methodol 2014;14:5
— and TRIPOD-Cluster asks which version you computed, so the docstring and the screen both say
"unweighted mean of per-day c-statistics" out loud.
Real rung-0 run: heat **0.728** within-day (0.844 pooled), treatment_gap 0.715 (0.744),
access_loss 0.645, mental 0.605, breathing 0.581 (0.581 — no day effect at all). 30/30 days
scoreable on every need, so no selection effect from dropping eventless days.

**The control, and the thing that answers it.** A single number at each need's base rate scores
**ECE 0.0000** — better than Leeward's 0.0011–0.0075. I first wrote this up as a binning
artifact and that was wrong: it is zero under *any* binning and any smoother, because
calibration error is proper but not *strictly* proper — it drops the sharpness term
(Gruber & Buettner, NeurIPS 2022). Austin & Steyerberg's ICI returns 0 here too. So no better
calibration metric rescues it, and the fix is a strictly proper score. `scaled_brier`
(1 − Brier/Brier_null, the Brier skill score / IPA, recommended by STRATOS TG6 2025) is now in
the table: the constant scores exactly 0.0 there, by construction.

**Read the skill column before the demo.** heat **+0.110**, treatment_gap **+0.042**,
breathing +0.001, mental **−0.004**, access_loss **−0.014**. Three of five needs are at or
below a constant at the base rate on a strictly proper score, while all three still rank above
chance (within-day AUC 0.58–0.65). Ranking and scale are different things: rung 0 has never
seen an outcome, so it orders people better than a coin while its probabilities are still the
priors' — and at a 0.34% base rate an ECE of 0.0067 is twice the base rate itself, which costs
more in squared error than the weak ranking earns back. **This is the clearest argument in the
repo for rung 1**, and it is on the screen rather than in a footnote. Do not quote heat alone.
PR-AUC is reported as secondary only: STRATOS TG6 and McDermott et al. (NeurIPS 2024) both
advise against preferring AUPRC to AUROC, the latter because it favours subpopulations with
more frequent positives — a fairness hazard in a project that ships a fairness audit.

`tests/test_guardrails.py` now fails if the control or within-day AUC stops reaching the
payload or the screen. The Model report is five sections, not four; `ReportResponse` gained
`constant_ece` and `discrimination` (additive/optional, cleared with the `api` owner first).
**Corrected `docs/ROUND_C.md`:** its AUC and PR-AUC columns reproduce exactly, but its two lift
columns are pooled over the window. Per-day selection — the same argument as within-day AUC —
gives heat 10.7x, not 18.2x; the pooled figure assumes a month of call budget banked for the
heat wave. Needs with no day effect are identical either way, which is what confirms it.

**The slider perf test, and what it was actually measuring.** Through this task
`test_api.py::test_the_slider_answers_inside_300ms_at_ten_thousand_veterans` was red at a
424 ms median (budget 300; its own comment records 127 ms idle) while the model lane ran
`leeward.model.fit --chains 4` at ~400% CPU and a cohort worktree ran `score_prior` at ~320%,
load average 22-66. It stayed red at a 404 ms median on a reading of 5.48 load average, which
looked like a refutation -- but that was the 1-minute figure decaying while the fit was still
finishing (5-min 17.4, 15-min 30.7) and the machine still thrashing. **Once both jobs actually
exited it passes, three runs out of three, at load average 16-29.** So it is contention, as the
import graph said it had to be: `leeward/api/**` has zero runtime imports of `leeward.eval`, so
nothing in this lane is reachable from POST /actions.
Worth writing down because the 1-minute load average lies on the way down: it read "quiet"
while a 400%-CPU job was still unwinding. Wait for the process to be gone, not for the number
to drop, before trusting any timing on this box.

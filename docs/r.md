# Round C — the prompts the live stack asked for

*Written 20 Sep 2026, ~00:15, against `main` at `a88f646` (PR #12, the UI redesign, merged
minutes earlier). Every number below was measured by running the stack, not read off a doc.*

`docs/PROMPTS.md` says Round C is rehearsal only. That was the right call when it was
written. Then the whole pipeline got run end to end for the first time, and five things
turned out to be true that nobody could have known without running it. Four of them touch
something on the never-cut list.

This file is the evidence, then the prompts, then the order. It does not replace
`docs/PROMPTS.md` — prompt **16 · `cohort` — honest missingness** is still open there and is
still the cheapest fix on the board. It is referenced below as C3 rather than restated.

**Read §1 before you paste anything.** Two of the five findings change what you say on
stage, not just what you build, and one of them makes a demo beat land that currently does
not.

---

## 0. What was verified, and how

| | |
| --- | --- |
| Gate | `make check` green at `a88f646` **with the real 10,000-veteran pipeline output in `data/`**, 2 skips (both the uncommitted Synthea zip). Prompt 3's "tests own their data" holds. |
| Pipeline | `make cohort` 1.8s · `score_prior` 9.3s (6,000,000 rows, 400 draws) · `allocate` 4.9s (312,281 actions over 106 days, total EHA 29,597.1) · `make report` 10.1s |
| API | All eight routes live. `POST /actions` 165–204 ms locally over 5 runs, 10,000 veterans. |
| UI | `npm run build` clean in 2.0s. All seven screens present. **Rendering still visually unverified** — the Chrome extension would not open a tab, third session running. Every route the UI calls was exercised against the live API instead; all return the shapes `ui/src/lib/types.ts` expects. |
| Unmerged | `lane/cohort` has prompt 7 done (race from ACS B03002, dialysis at VA ESRD prevalence) and unmerged. Its red CI is **staleness only** — it predates `8d79891` so it lacks `budget = 3000 if CI else 300` in `tests/test_api.py`. Merge main into it and it goes green. |

Somebody still has to look at those seven screens with their eyes before you stand up. No
test in this file substitutes for that.

---

## 1. The five findings

### 1.1 The capacity slider moves the counter by 4.4%

The slider is on the never-cut list and it is described as the best three seconds of the
demo. Measured on Sandy landfall day, 2026-08-03, 10,000 veterans:

| calls | `total_eha` |
| --- | --- |
| 10 | 729.2 |
| 20 | 735.0 |
| 40 | 747.3 |
| 80 | 767.2 |
| 100 | 776.6 |

20 → 80 is **+4.4%**. The mechanism is right; the display buries it. Here is where the
747.3 comes from:

| bucket | n | EHA | % of total |
| --- | --- | --- | --- |
| **free** (verified_text, capacity 10,000) | 5,978 | 482.7 | **64.6%** |
| refill | 200 | 140.6 | 18.8% |
| **call** ← the slider | 40 | 34.1 | **4.6%** |
| booking | 20 | 30.8 | 4.1% |
| va_fill | 30 | 25.0 | 3.3% |
| ride | 15 | 17.2 | 2.3% |
| evac | 8 | 10.4 | 1.4% |
| pharmacist_slot | 12 | 6.6 | 0.9% |

Two-thirds of the headline number is a free text that every strategy gets for nothing. The
same dilution flattens the lift:

```
leeward          747.3
rank_by_chronic  595.3
random           591.4
rank_by_age      587.8      -> 1.26x as displayed
                            -> 2.35x with the free bucket removed from both sides
```

**The decision layer is roughly twice as good as the screen says it is.** Fix is
presentational, not algorithmic. → **C2**

### 1.2 The calibration plot cannot beat a constant

A single number equal to the base rate, scored on the same held-out window (30 days,
1,500,000 pairs), with `calibration.py`'s own equal-mass binning:

| need | base rate | Leeward ECE | constant ECE |
| --- | --- | --- | --- |
| breathing | 0.67% | 0.0011 | **0.0000** |
| heat | 2.33% | 0.0075 | **0.0000** |
| mental | 0.50% | 0.0044 | **0.0000** |
| treatment_gap | 1.82% | 0.0050 | **0.0000** |
| access_loss | 0.34% | 0.0067 | **0.0000** |

The equal-mass binning argument in that module's docstring is correct and well made. It
still cannot separate the model from one that knows nothing about anyone, because at these
base rates calibration is not the discriminating question.

The discrimination is there, and **nothing in the repo reports it**:

| need | within-day AUC | pooled AUC | lift@1% | lift@10% | PR-AUC |
| --- | --- | --- | --- | --- | --- |
| heat | **0.728** | 0.844 | 18.2× | 6.0× | 0.241 |
| treatment_gap | **0.707** | 0.734 | 19.3× | 4.2× | 0.159 |
| access_loss | **0.646** | 0.653 | 4.4× | 2.7× | 0.008 |
| mental | **0.605** | 0.613 | 2.8× | 2.1× | 0.008 |
| breathing | **0.581** | 0.581 | 3.3× | 2.0× | 0.011 |

Within-day is the honest column: the call list is chosen within a day, so pooled AUC is
partly "was today a heat wave", which is not a decision. Heat drops 0.844 → 0.728 when you
take the day effect out. That is still a real, defensible, prior-only number.

[TRIPOD+AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/) — the 2024 reporting
standard for clinical prediction models — names three things: **discrimination,
calibration, clinical utility**. Leeward reports calibration and clinical utility. The
missing leg is ~50 lines against pairs `calibration.paired()` already builds. → **C5**

### 1.3 `find_out` fires on 4 of 6,303 actions

On landfall day: `act_now` 339, **`find_out` 4**, `self_serve` 5,960, `everyday` 0. And
`score_prior` prints the cause directly — `epistemic share >= 0.4 on 0.0%` of rows, for
every one of the five needs.

"A wide interval earns a cheap three-minute check-in call" is one of the five things the
pitch is built on. It fires on 0.06% of actions because `leeward/cohort/missingness.py` was
never built, so all 10,000 veterans have complete data and epistemic variance is flat.

This is already written up as prompt **16 · `cohort`** in `docs/PROMPTS.md`. It is still
the cheapest certain win on the board. → **C3**

### 1.4 The four rules that are the four stories are not implemented

`leeward/decision/tiers.py`, verbatim from its own docstring:

> Not yet implemented: the hazard-triggered act-now rules (site-dependent × SiteDown; no
> caregiver + powered equipment + outage; mail-order supply short on a delivery-disrupted
> day; controlled substance × SiteDown).

Those four rules are, in order: the Sandy dialysis/OTP story, the emPOWER story, the 80%-
mail-order story, and the controlled-substance story. All four are rule-based and need no
posterior — `docs/BUILD_PLAN.md` §6 cut item 8 explicitly protects exactly this.

Related, and still unresolved from prompt 10: on landfall day the 40 scarce call slots go
**21 to `act_now`, 19 to `self_serve`**, and **234 of 255 `act_now` veterans get no call at
all** — they get a `verified_text`. A judge who reads the tier badge "Act now → call today,
VA care team" and then finds an automated text is a bad ninety seconds. → **C4**

### 1.5 The offline claim is not proven for the UI

`scripts/clean_clone_test.sh` clones, makes a venv, runs `make_fixtures.py`, runs pytest.
It never boots the API and never touches `ui/`. `scripts/smoke_demo.sh` does block the
network properly and hit every route — but also never builds or serves the UI.

`node_modules/` and `ui/dist/` are both in `.gitignore` and neither is tracked. **A
genuinely offline clean clone cannot `npm ci`.** `make demo` works on Rahul's laptop
because `node_modules` is already sitting there.

`docs/BUILD_PLAN.md` §10 says the T+21:00 checkpoint is "clean-clone test passes offline in
under 60 s". Today that sentence is not true and nothing catches it. → **C6**

---

## 2. Smaller things that are still real

| | |
| --- | --- |
| **`make score` and `make fit` both fail** | No `leeward.model.score`, no `leeward.model.fit`. `scripts/baseline.py` has the working stage list (`score_prior`, then `allocate`). Six places in the docs — including the Merge-1 and Merge-2 rituals in `PROMPTS.md` — tell someone to run a broken target. → **C1** |
| **`ablations: []`** | `ablate.py` was never built, so the Report screen carries an empty card. "Does the SiteDown term matter?" is a never-cut claim you currently cannot evidence. → **C7** |
| **Fairness cannot flag** | FNR is 0.93–0.99 in *every* group, because 40 calls cannot cover 16,443 events. Two numbers both at the ceiling cannot diverge 20%. Worst ratio across 29 groups is 1.035. → **C8** |
| **…and it is hiding a good finding** | low income FNR 0.885 vs high 0.981 · HVI-5 0.928 vs HVI-1 0.989 · no caregiver 0.927 vs coresident 0.982. **Leeward reaches the most vulnerable more often.** Nothing says so, because everything renders "not flagged". → **C8** |
| **race / ethnicity are one "unknown" group** | ratio 1.0 against itself, on the screen that is supposed to answer NYC Health's 3× Black/white heat-stress disparity. Fixed on `lane/cohort`, unmerged. → **C0** |
| **One scenario is demoable** | `leeward/api/main.py` serves one cached hazards table and 422s the rest. The smoke replay (AQI 254, 7 Jun 2023) is your most checkable beat and cannot be shown. → **C10**, expect to cut |
| **`FacilityStatus` has no date** | so a closure renders as a window-level strip, not on the day it starts. Your own `status/ui.md` flagged it. → **C9** |
| **No `n_not_reached`** | the week board pays 7 extra uncapped `POST /actions` to compute "14 veterans not reached". → **C9** |
| **`/actions` returns 6,303 rows for one day** | 1,497 → 7,802 across the seven-day window. The board needs a top-N. → **C9** |
| **`headline` is not in the contract** | it rides on `allow_extra=True`, so `schema.empty("actions")` lacks it and `ActionRow.headline` is `str \| None`. Prompt 14 preferred a real column. Low priority, but write it down. |

---

## 3. What the web research changed

Four things worth putting in the deck, and one that reframes the whole pitch.

**The 82 °F hinge is stronger than the deck says.** [NYC Health's 2026
report](https://www.nyc.gov/site/doh/about/press/pr2026/nyc-health-department-releases-report-on-heat-related-deaths.page):
**~80% of heat-related deaths happen at 82–94 °F**, not on extreme-heat days — and
**non-extreme hot days have doubled, ~14 → ~32 per summer, over five decades.** The second
number turns a hyperparameter into a thesis. Neither is in `docs/sources.md` yet.

**REACH VET is the precedent, and it is the answer to "would VA ever deploy this?"**
[VHA already runs model → dashboard → coordinator → call
nationally](https://psychiatryonline.org/doi/full/10.1176/appi.ps.202100629): 61 EHR
variables, monthly, top 0.1% risk stratum (30× the base suicide rate), evaluated to produce
more safety plans and fewer ED visits. **Leeward is REACH VET for climate** — daily instead
of monthly, hazard-triggered instead of scheduled, multi-need, and capacity-aware.

**emPOWER has already been used this way, in NYC.** [ASTHO's
brief](https://www.astho.org/topic/brief/leveraging-the-hhs-empower-program-to-enhance-power-outage-planning/):
the city bought phone numbers for electricity-dependent Medicare beneficiaries and sent
automated pre-storm messages; Broome County identified 58 and called all 58. That is the
capacity argument in one line — **calling all 58 works, calling all 10,000 does not.**

**The sharpest rule checks out.** The [VA Pharmacy Disaster Relief
Plan](https://www.va.gov/fayetteville-coastal-health-care/programs/pharmacy-disaster-relief-plan/)
gives a 10-day emergency supply at in-network community pharmacies, **controlled substances
excluded**. Name the program in the action text, not just the mechanism.

**A facility-side citation that is not Sandy.** [VA's 2024–2027 Climate Adaptation
Plan](https://www.sustainability.gov/pdfs/va-2024-cap.pdf) scores facility heat exposure —
15% of VHA facilities high, ~half moderate — and names grid failure as the key operational
risk.

---

## 4. The prompts

House rules unchanged. One prompt per terminal. Each is self-contained. Each ends the same
way, and that ending is the point:

> Run `make check`. It must be green. Then commit, push, write two lines to
> `status/<lane>.md`, and stop. Do not start the next task.

### Order

| | Lane | Prompt | Owner | Est. |
| --- | --- | --- | --- | --- |
| **Tonight** | — | **C0 · merge chores** (not a prompt — do it by hand) | either | 10 min |
| | `api` | C1 · `make score` and `make fit` tell the truth | Rahul | 10 min |
| | `ui` | C2 · the counter shows what the slider buys | Rahul | 30 min |
| | — | **Record the backup video.** Overdue since T+9:30. | Rahul | 20 min |
| | | *then sleep — a rehearsed demo beats an extra feature* | | |
| **Morning** | `api` | C4 · the four hazard-triggered act-now rules, and prompt 10 | Rahul | 2 h |
| | `cohort` | C3 · honest missingness (= PROMPTS.md prompt 16) | Harshini | 1 h |
| | `eval` | C5 · discrimination, and the constant-predictor control | Harshini | 1.5 h |
| | `demo` | C6 · prove the UI boots offline | Rahul | 1 h |
| **Then** | `eval` | C7 · `ablate.py` — does SiteDown matter? | Harshini | 1 h |
| | `eval` | C8 · the fairness audit reports the ratio and the good news | Harshini | 45 min |
| | `api` | C9 · three contract gaps the week board is working around | Rahul | 45 min |
| **Last** | `model` | C10 · rung 1, time-boxed to the toy cohort | Harshini | 90 min |
| *cut first* | `demo` | C11 · make the smoke replay demoable | Rahul | 90 min |

**Cut order, no discussion:** C11, then C10, then C9, then C8. Everything above C8 is
load-bearing for a beat that is on the never-cut list.

---

### C0 · merge chores — do these by hand, now

Not a prompt. Four commands and a look.

```bash
# 1. lane/cohort is green the moment it stops being stale
git checkout lane/cohort && git merge --no-ff main && make check && git push

# 2. take it, and prompt 7 with it — this is what the fairness screen is waiting for
git checkout main && git merge --no-ff lane/cohort && make check && git push

# 3. everything else is behind a88f646
git -C ../Climate-Health-Cornell-Hackathon pull
git checkout lane/ui && git merge --no-ff main
```

`lane/cohort`'s CI is red for one reason: it predates `8d79891`, so `tests/test_api.py`
still hard-codes a 300 ms budget instead of `budget = 3000 if os.environ.get("CI") else
300`. GitHub's runners take 390–420 ms. Nothing is wrong with the code.

---

### C1 · `api` — `make score` and `make fit` tell the truth

`make score` runs `python -m leeward.model.score`. That module does not exist. Neither does
`leeward.model.fit`, which `make fit` runs. Both fail immediately. `scripts/baseline.py`
has the stage list that actually works — `leeward.model.score_prior`, then
`leeward.decision.allocate` — so the Makefile and the baseline recorder disagree about what
scoring is, and the Makefile is the one in six places of documentation.

> Read `scripts/baseline.py` (the `STAGES` list) and the `score` and `fit` targets in
> `Makefile`.
>
> Point `make score` at the stages that exist: `leeward.model.score_prior` then
> `leeward.decision.allocate`. Make `make fit` exit non-zero with one sentence naming the
> rung that is built and the module that would have to land for a fit to happen — a target
> that fails with an explanation is fine, a target that fails with `No module named` is not.
>
> When `leeward/model/fit.py` lands (C10), `make fit` starts working with no further edit,
> and `make score` picks up `score.py` over `score_prior.py` if it is there.
>
> **Write the test first**: a test that runs each documented Makefile target's command line
> and asserts the module is importable, so the next broken target is caught by the gate
> rather than by a person at 2am. `make demo`, which blocks, is the one exception — assert
> its modules import, do not run it.
>
> Then fix the six references in `docs/PROMPTS.md`, `docs/BUILD_PLAN.md` and `README.md` if
> and only if they are now wrong.
>
> Run `make check`; green; commit; push; two lines in `status/api.md`; stop.

---

### C2 · `ui` — the counter shows what the slider buys

The capacity slider is on the never-cut list and `docs/BUILD_PLAN.md` §8 calls it "the
single most persuasive three seconds you have". Measured on landfall day, moving it from 20
calls to 80 moves the counter from **735.0 to 767.2 — 4.4%.**

The allocator is not wrong. 64.6% of `total_eha` is the `free` bucket — 5,978 verified
texts at capacity 10,000, which every strategy receives whatever it does. The `call` bucket
the slider controls is 4.6% of the total. The counter is measuring mostly the part that
does not move.

The same dilution is in the lift: `/actions` reports Leeward 747.3 against rank_by_age
587.8 — 1.26×. Remove the free bucket from both sides and it is **2.35×**.

> Read `ui/src/components/HarmCounter.tsx`, `ui/src/components/CapacitySlider.tsx`,
> `ui/src/screens/CareTeam.tsx` and `leeward/decision/allocate.py` (`total_eha`, and
> `schema.ACTION_COST_UNIT` for which bucket an action spends).
>
> Make the counter the slider is next to report **harm averted by the scarce slots** — the
> actions whose `capacity_bucket` is not `free` — with the free-text floor shown once,
> separately, as the thing the team gets without spending anyone's afternoon. Two numbers
> with two labels, not one number that is secretly both.
>
> Keep `total_eha` on screen somewhere, labelled, because it is what `/actions` returns and
> a judge may ask. This is a display decision: **do not change the allocator and do not
> change what the API returns.**
>
> Apply the same split to the baseline comparison so the displayed lift is computed on the
> same basis as the displayed counter. The two EHA labels (`EHA_EXPECTED` /
> `EHA_REALIZED`) that `demo/opening-day` added stay exactly as they are — this is a third
> axis, not a replacement.
>
> **Acceptance, and write it as a test against the fixtures first:** moving capacity from
> 20 to 80 calls changes the displayed scarce-slot number by more than 50%, where today it
> changes the displayed total by 4.4%. `npm run build` clean.
>
> Run `make check`; green; commit; push; two lines in `status/ui.md`; stop.

---

### C3 · `cohort` — honest missingness

**This is prompt 16 · `cohort` in `docs/PROMPTS.md`, already written. Paste it from there:
`make prompt N=16`.** Do not re-specify it here.

The live numbers that confirm it, for whoever wants them: on landfall day the tier mix is
`act_now` 339, **`find_out` 4**, `self_serve` 5,960, `everyday` 0 — 0.06% of 6,303 actions.
`score_prior` prints `epistemic share >= 0.4 on 0.0%` for all five needs. Nobody is ever
uncertain enough to earn a check-in because nobody is ever missing anything.

---

### C4 · `api` — the four hazard-triggered act-now rules, and the prompt-10 decision

Two things, one lane, because the second is much easier once the first lands.

**First.** `leeward/decision/tiers.py` says so itself:

> Not yet implemented: the hazard-triggered act-now rules (site-dependent × SiteDown; no
> caregiver + powered equipment + outage; mail-order supply short on a delivery-disrupted
> day; controlled substance × SiteDown). They need hazards and site_status, which
> `allocate()` does not take yet.

Those four rules are the four stories the deck is made of. All four are rule-based and need
no posterior; `docs/BUILD_PLAN.md` §6 cut item 8 protects exactly this. The cohort already
carries every column they need — `med_controlled` fires on 5.5% of the panel (547 veterans),
`med_cold_chain` on 10.8%, `mail_order_pharmacy` on ~80%, and 107 veterans are on OTP with
81 of them at station 630, which Sandy closes for 45 days.

**Second.** Prompt 10 has never been decided and the symptom is now measured on real data:
on landfall day the 40 scarce call slots go **21 to `act_now` and 19 to `self_serve`**, and
**234 of the 255 `act_now` veterans get no call** — only a `verified_text`. The tier badge
says "Act now → call today, VA care team".

> Read `leeward/decision/tiers.py`, `leeward/decision/allocate.py`, `docs/SPEC.md` §7.5 and
> `docs/proposal.md` §6.
>
> **Part 1.** Give `assign()` the hazards and site_status it needs and implement the four
> hazard-triggered act-now rules named in its own docstring. A veteran who matches one is
> `act_now` regardless of where their probability sits, because the rule encodes a mechanism
> the probability has not seen. Put the four rules in a table a clinician could read, beside
> `severity.yaml` and `tau.yaml`, not buried in a `when/then` chain.
>
> The mechanisms, with their sources, so you do not have to re-derive them:
> - **site-dependent × SiteDown** — dialysis, infusion and OTP at a closed station.
>   Station 630 sits in evacuation zone 1 and closed for five months after Sandy;
>   ~100 veterans needed emergency guest-dosing.
> - **controlled substance × SiteDown** — the VA Pharmacy Disaster Relief Plan gives a
>   10-day retail supply and **excludes controlled substances**, so the fallback that covers
>   everyone else does not exist for these veterans. Cite the program by name in the
>   rationale.
> - **mail-order supply short on a delivery-disrupted day** — ~80% of VA outpatient
>   prescriptions arrive by mail, so a flooded ZIP is a medication-supply event.
> - **no caregiver + powered equipment + outage** — the emPOWER rationale.
>
> **Part 2.** Implement resolution **A** of prompt 10: a tier priority band in the
> allocation order, so a scarce bucket fills `act_now` before `find_out` before
> `self_serve`, EHA-ranked within each band. Then update `docs/SPEC.md` §7.5 to say that,
> because today it does not.
>
> **Write the tests first.** (a) A dialysis patient of station 630 on a day 630 is down is
> `act_now` even when their `p_mean` is under 0.25. (b) A veteran on a controlled substance
> at a closed site is `act_now` and their rationale names the Pharmacy Disaster Relief Plan.
> (c) After Part 2, no `self_serve` veteran holds a scarce-bucket slot while an `act_now`
> veteran who could use that bucket holds none. (d) Total EHA does not fall by more than 15%
> versus today — record today's number before you start.
>
> `test_guardrails.py::test_more_capacity_never_averts_less_harm` must stay green; the tier
> band must not break capacity monotonicity. You own `leeward/api/schemas.py` — tell the
> other terminals if you change it.
>
> Run `make check`; green; commit; push; two lines in `status/api.md`; stop.

---

### C5 · `eval` — discrimination, and the constant-predictor control

The calibration plot is on the never-cut list. Measured against a control nobody had run: a
single number equal to the base rate, scored on the same 30-day held-out window with
`calibration.py`'s own equal-mass bins, gets **ECE 0.0000 on all five needs** — strictly
better than Leeward's 0.0011–0.0075. At base rates of 0.34%–2.33%, calibration is not the
discriminating question, and the flagship science screen currently cannot tell the model
apart from one that has never met anyone.

The discrimination is there. It is just not reported anywhere in the repo:

```
need            within-day AUC   pooled AUC   lift@1%   PR-AUC
heat                     0.728        0.844     18.2x    0.241
treatment_gap            0.707        0.734     19.3x    0.159
access_loss              0.646        0.653      4.4x    0.008
mental                   0.605        0.613      2.8x    0.008
breathing                0.581        0.581      3.3x    0.011
```

Within-day is the number that justifies a call list — the list is chosen within a day, so
pooled AUC is partly "was today a heat wave", which is not a decision anyone makes.
[TRIPOD+AI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11019967/), the 2024 reporting standard
for clinical prediction models, names discrimination, calibration and clinical utility.
Leeward reports two of the three.

> Read `leeward/eval/calibration.py` — especially `paired()` and `holdout_dates()`, which
> already build exactly the frame this needs — and `leeward/eval/report.py`.
>
> Implement `leeward/eval/discrimination.py`: per need, on the held-out window, **within-day
> AUC** (the mean over days of that day's AUC, and say in the docstring why it is not the
> pooled one), pooled AUC beside it, PR-AUC, and lift at the top 1% and top 10%. Use the
> rank identity with tie-averaging; do not add a scikit-learn dependency for it.
>
> Then add the **constant-predictor control** to `calibration.py`'s output: one row per need
> giving the ECE a single number at the base rate scores on the same bins. Render it on the
> reliability chart as a labelled reference line. This is the honest half of the calibration
> claim and it is more persuasive than hiding it, because the answer to "what would a
> constant score?" is a number you already know rather than a pause.
>
> Extend `ReportResponse` in `leeward/api/schemas.py` — **it is the `api` lane's file, so
> ask before you touch it**; the owner pushes within five minutes. Then surface both on
> `ui/src/screens/Report.tsx`.
>
> **Write the test first.** Within-day AUC for `heat` is above 0.65 and below the pooled
> value; a perfectly-ranked synthetic predictor scores AUC 1.0 and a shuffled one scores
> 0.5 ± 0.02; the constant-predictor ECE is at or below the model's on every need, which is
> the finding, so the test asserts it rather than hoping.
>
> Run `make check`; green; commit; push; two lines in `status/eval.md`; stop.

---

### C6 · `demo` — prove the UI boots offline, because today nothing does

`scripts/clean_clone_test.sh` clones, makes a venv, runs `make_fixtures.py` and runs pytest.
It never boots the API and never touches `ui/`. `scripts/smoke_demo.sh` blocks the network
properly and hits every route, but also never builds or serves the UI.

`node_modules/` and `ui/dist/` are both in `.gitignore` and neither is tracked, so **a clean
clone on a machine with the wifi off cannot `npm ci` and therefore cannot run `make demo`.**
It works on the demo laptop because `node_modules` is already there. `docs/BUILD_PLAN.md`
§10 says the T+21:00 checkpoint is "clean-clone test passes offline in under 60 s"; that
sentence is not true today and nothing catches it.

> Read `scripts/clean_clone_test.sh`, `scripts/smoke_demo.sh`, the `demo` target in
> `Makefile`, and `leeward/api/main.py`.
>
> Commit the built `ui/dist/` and serve it from FastAPI as static files at `/`, so the demo
> path is one process with no dev server, no HMR and no cold Vite start. Keep `npm run dev`
> working for development; `make demo` should prefer the built bundle and say which one it
> booted. Un-ignore `ui/dist/` and add a `make ui` that rebuilds it, so it is obvious the
> committed artifact is regenerable.
>
> Then make `clean_clone_test.sh` test the thing it claims: clone, install, boot, **curl the
> served page and assert it contains the app root**, hit one real route, and print the
> wall-clock seconds. Keep the network blocked the way `smoke_demo.sh` already does, with
> the proxy env vars, so a stray fetch fails inside the test rather than on stage.
>
> **Acceptance:** the script passes end to end with wifi physically off, in under 60
> seconds, and fails loudly if `ui/dist` is missing or stale relative to `ui/src`. Write the
> measured boot time into `status/demo.md` — that is the number `BUILD_PLAN` §10 asks for.
>
> Run `make check`; green; commit; push; two lines in `status/demo.md`; stop.

---

### C7 · `eval` — `ablate.py`, because "does the SiteDown term matter?" has no answer

`GET /report` returns `ablations: []`. `leeward/eval/report.py` says so in a comment — "an
empty list means not run" — which is honest, and leaves an empty card on the science screen.
Prompt 12 specified an ablation table and the screen has a slot for one.

The SiteDown term is on the never-cut list. Right now the evidence for it is a story about
Sandy and a join against the evacuation-zone table. An ablation makes it a measurement, and
at rung 0 an ablation is cheap: zero a coefficient block in the design, re-score, re-measure.
No refit, no MCMC.

> Read `leeward/model/design.py` (`coef_matrix` and the term layout),
> `leeward/model/score_prior.py`, `leeward/eval/decision_quality.py` and the `ablations`
> stub in `leeward/eval/report.py`. `AblationRow` is already in `leeward/api/schemas.py`.
>
> Implement `leeward/eval/ablate.py`: for each of a named set of terms, zero that term's
> block, re-score the held-out window, and report the change in harm averted at 40 calls and
> in ECE. Ablate at least `psi_sitedown`, `theta_sitedown_x_sitedependent`, the medication
> block, and the heat lag — those are the four the deck makes claims about.
>
> Seed it and make it re-runnable in under two minutes on the real cohort, because it will
> be run again after C3 and C4 land and the numbers will move.
>
> **Write the test first.** Zeroing `psi_sitedown` and `theta_sitedown_x_sitedependent`
> strictly lowers harm averted on the days station 630 is down, and changes nothing on days
> no site is down — the second half is the one that catches a broken ablation, so do not
> skip it.
>
> Run `make check`; green; commit; push; two lines in `status/eval.md`; stop.

---

### C8 · `eval` — the fairness audit reports the ratio, and the good news

The audit runs, renders, and flags nothing — 0 of 29 groups, worst relative FNR gap 1.035.
That reads as a pass. It is closer to a metric that cannot fail: **FNR is 0.93–0.99 in every
single group**, because 40 calls a day cannot cover 16,443 events in a 30-day window. Two
numbers pinned at the ceiling cannot diverge by 20%.

Meanwhile the audit is sitting on a finding nobody is saying:

```
income_band   low  FNR 0.885   vs high 0.981
hvi_band      HVI5 FNR 0.928   vs HVI1 0.989
caregiver     none FNR 0.927   vs informal_coresident 0.982
medication    5-9 meds 0.915   vs 0-4 meds 0.991
```

**Leeward reaches the most vulnerable more often, on four independent axes.** That is the
fairness result, and the screen renders it as four rows of "not flagged".

> Read `leeward/eval/fairness.py` and `report/fairness.csv`.
>
> Keep raw FNR in the CSV — it is the honest denominator and removing it would be
> suppression. But make the reported and rendered quantity the one that can move: the **FNR
> ratio to cohort**, which is already computed, plus **coverage at a fixed call budget by
> group** (what share of that group's events got a scarce action), which is the thing a care
> team can actually change. Say on the screen, in one line, that FNR is near 1 for everyone
> because the budget is 40 calls against 16,443 events — otherwise a judge reads 0.95 and
> concludes the model does not work.
>
> Then surface the direction. A group reached **more** than the cohort is a result, not an
> absence of a problem; mark it as such and let `ui/src/screens/Report.tsx` render it.
> The 20% relative-gap bar stays exactly where it is, and flagged rows stay visually marked —
> **a failing audit is displayed, never suppressed**, and `test_guardrails.py` enforces the
> backend half.
>
> **Write the test first.** A synthetic cohort where one group is deliberately under-reached
> must flag; the same cohort with the groups swapped must flag the other one; and a group
> reached *more* than the cohort must not flag but must be visible in the output.
>
> This depends on `lane/cohort`'s race and ethnicity work being merged (C0) — until then
> those two strata are a single "unknown" group with a ratio of 1.0 against itself.
>
> Run `make check`; green; commit; push; two lines in `status/eval.md`; stop.

---

### C9 · `api` — three contract gaps the week board is working around

All three are in `status/ui.md` already. The board ships despite them; each one costs it
something.

1. **`FacilityStatus` carries no date**, so a closure renders as a window-level strip
   instead of landing on the day it starts. Sandy closes station 630 for 45 days and the
   ribbon cannot show which day that begins.
2. **`ActionsResponse` has no `n_not_reached`**, so the board issues seven *extra*
   uncapped `POST /actions` calls to compute "14 veterans not reached at this capacity" —
   fourteen requests to draw seven numbers.
3. **`POST /actions` returns every row** — 1,497 to 7,802 per day across the opening window,
   6,303 on landfall day, almost all of them free verified texts. The board renders a top
   slice.

> Read `leeward/api/schemas.py` (`FacilityStatus`, `ActionsRequest`, `ActionsResponse`),
> `leeward/api/main.py` and `ui/src/lib/api.ts` (`postActionsWeek`). You own
> `api/schemas.py`; tell the `ui` terminal when each lands, they are independent.
>
> Add a `date` to `FacilityStatus` so a closure is a per-day fact. Add `n_not_reached` to
> `ActionsResponse` — computed server-side where the uncapped allocation is nearly free,
> rather than by a second round trip. Add an optional `limit` to `ActionsRequest`, defaulting
> to no limit so nothing already working changes, with `n_selected` still reporting the true
> total so the board can say "40 of 6,303".
>
> **Write the test first** for each: a closure that starts mid-window appears on its own day
> and not before it; `n_not_reached` equals what the uncapped call returns, so the board can
> stop making it; `limit` changes the rows returned and does not change `n_selected`,
> `total_eha` or `counts_by_tier`.
>
> Then tell the `ui` terminal, so the ribbon can place the closure and `postActionsWeek` can
> drop from 14 requests to 7.
>
> Run `make check`; green; commit; push; two lines in `status/api.md`; stop.

---

### C10 · `model` — rung 1, time-boxed to the toy cohort

**Read this framing before deciding to run it.** `docs/PROMPTS.md` says to drop rung 1
before dropping prompts 1, 2 or 4, and that advice still holds: rung 0 already gives a
within-day AUC of 0.728 on heat, `recovery.py` is scrupulous about labelling a prior-only
interval as "do the priors bracket the truth" rather than recovery, and `data/truth.json`
was deliberately planted *away* from the prior means so a fit has somewhere to move.

But the project is called a Bayesian daily-hazard model and there is no MCMC anywhere in it.
At an AI hackathon that is a question you will be asked. So: **do not attempt the full fit.**
Buy the answer to the question, not the upside.

> Read `docs/SPEC.md` §6 and §6.0 (the ladder), `leeward/model/priors.py`,
> `leeward/model/design.py` (**do not fork it, do not change it**) and
> `leeward/eval/recovery.py` — especially `SHAPE_NOTE`, which states the exact array layout
> this module will accept from you.
>
> Implement `leeward/model/hazard.py` and `leeward/model/fit.py` for **rung 1 only**:
> binomial-cell likelihood, five needs, `alpha`, the health betas and the 4-lag heat curve.
> **No ICAR, no latent dose, no interactions** — leave them as TODOs behind a flag.
>
> **Fit the 200-veteran, 30-day toy cohort first and treat that as the deliverable.** SPEC
> §6 says it completes in under 60 seconds with zero post-warmup divergences. Write
> `data/posterior.nc` as ArviZ InferenceData with variable names matching `priors.py`, and
> write r-hat and the divergence count into `report/fit.json`.
>
> **Stop there if the clock says stop.** The full 10,000-veteran fit is a nice-to-have; a
> toy fit with a real r-hat you can say out loud is the whole point. Say which one ran.
>
> When it lands, `recovery.py` reads `posterior.nc` instead of drawing priors and its claim
> becomes parameter recovery proper, with no change to that module. `make fit` (see C1)
> starts working with no change to the Makefile.
>
> **Write the test first**: the toy fit completes in under 60 s with zero post-warmup
> divergences and r-hat < 1.05 on every parameter; `posterior.nc` loads through
> `recovery.py` without a shape error; `model_rung` is 1 in the scores it produces.
>
> Run `make check`; green; commit; push; two lines in `status/model.md`; stop.

---

### C11 · `demo` — make the smoke replay demoable ← cut this first

`leeward/api/main.py` serves one cached hazards table and returns 422 for any other
scenario. `scenarios/` holds three. `ida_flash_flood` is already cut list item 6, but the
smoke replay is not, and it is the single most checkable thing in the build: real EPA AirNow
monitor data, peak PM2.5 203.5 µg/m³ and AQI 254 on a Queens monitor, 7 June 2023. A judge
can look it up on their phone.

Today it cannot be shown, because there are no scores behind it.

> Read `leeward/api/store.py`, `leeward/api/main.py` (`SCENARIO`, `_check_scenario`) and
> `leeward/ingest/hazards.py`.
>
> Give the cached tables a scenario dimension — `data/<scenario>/*.parquet` — and let the
> store hold more than one. Score `smoke_2023` as well as `sandy_then_heat` and commit
> nothing new to `data/` that a `make` target cannot rebuild. Make `_check_scenario` accept
> any scenario that has cached tables and keep the 422 for the ones that do not.
>
> Then put the scenario picker beside the day picker in the topbar, so "here is Sandy, and
> here is the week New York breathed Canadian wildfire smoke" is one click.
>
> **Acceptance:** `GET /forecast?scenario=smoke_2023` returns the 5–11 June 2023 window with
> a PM2.5 peak matching `airnow_pm25_nyc_smoke2023.parquet` to the decimal; `make demo`
> still boots in under 60 s; `make check` green.
>
> **If this is not done and merged by the freeze, cut it and say nothing on stage about
> three scenarios.** Do not demo a scenario picker with one working entry.
>
> Run `make check`; green; commit; push; two lines in `status/demo.md`; stop.

---

## 5. Deck edits, which are Rahul's and not an agent's

Five lines, none of which need code:

1. **82 °F becomes a thesis.** "~80% of NYC's heat deaths happen between 82 and 94 °F — and
   those days have doubled, 14 to 32 a summer, over five decades." Add both to
   `docs/sources.md` with the 2026 report URL.
2. **Open on REACH VET.** "VA already runs a risk model into a call list — 61 EHR variables,
   monthly, top 0.1%. Leeward is that, for climate: daily, hazard-triggered, capacity-aware."
   It answers "would VA ever deploy this?" before it is asked.
3. **emPOWER is the capacity argument.** NYC already sent automated pre-storm messages to
   electricity-dependent beneficiaries; Broome County identified 58 and called all 58.
   Calling all 58 works. Calling all 10,000 does not. That is the product.
4. **Name the Pharmacy Disaster Relief Plan.** Not "retail refill excludes controlled
   substances" — the program has a name and the exclusion is published.
5. **Say the fairness result out loud** (after C8): "the audit found no group under-reached,
   and found we reach low-income, high-HVI, no-caregiver and high-medication veterans *more*
   than the cohort." Then say what the audit cannot yet see, and why.

And the standing one from `BUILD_PLAN` §8: **know your rung.** Today the honest sentence is
"rung 0, prior-only, no MCMC — the ladder is in the repo and rung 1 is behind a flag." After
C10 it becomes "we fit rung 1 on a toy cohort, r-hat X, zero divergences; the full fit did
not finish." Either is a good answer. Silence is not.
# Prompt library

Copy-paste, one per Claude Code session. Each is self-contained: an agent starting in a
fresh worktree with no memory of this conversation can execute it.

**If you are not reading the code, this file is the code review.** A prompt that does not
state its acceptance test is a prompt that will produce something plausible and wrong. Every
prompt below ends the same way, and that ending is the point:

> Run `make check`. It must be green. Then commit, push, write two lines to
> `status/<lane>.md`, and stop. Do not start the next task.

---

## How this works

```bash
make setup          # once
make lanes          # creates the six worktrees
make fixtures       # correctly-shaped fake data for every table
make check          # THE GATE. Green = safe to merge. Red = not.
make status         # where the build is, without opening a file
```

**The gate is `make check`, not a person.** Lint, every test, and the guardrails in
`tests/test_guardrails.py` — which check that numbers *mean* the right thing, not just that
code ran. Capacity is never exceeded. Every message carries the anti-scam elements. Synthetic
columns carry their flag. The demo makes no network calls. Ranks are ordered by EHA. Nothing
is randomised without a seed.

**Guardrails for unbuilt modules skip with a message naming what they will enforce, and start
enforcing the moment that module lands.** So `make check`'s skip list is an accurate, live
to-do list, and the suite gets stricter on its own as lanes land.

**Handoff is a file, not a conversation.** Each lane appends to `status/<lane>.md` — separate
files, so six agents never conflict. `make status` reads the repo, not your memory.

**Merge:** in the lane, `make check` green → commit → push. On `main`, `git merge --no-ff
lane/<x>` → `make check` → push. Every 60–90 minutes, even if incomplete.

---

## What to run right now

Wave 1 is merged. Twelve prompts remain, in two rounds of six. **One prompt per terminal.**
Paste all six of a round at once; they are independent by construction.

Don't hunt through this file to copy one — `make prompt N=4`, or
`bash scripts/prompt.sh 4 | pbcopy` to put it straight on the clipboard.
`make prompt` with no argument lists all twelve.

| | Terminal | Prompt | Owner |
| --- | --- | --- | --- |
| **Round A** | `cohort` | 1 · medications | Harshini |
| | `model` | 2 · the simulator | Harshini |
| | `eval` | 3 · tests own their data | Harshini |
| | `api` | 4 · the FastAPI app | Rahul |
| | `demo` | 5 · verified messages | Rahul |
| | `ui` | 6 · veteran card + message screen | Rahul |
| **Round B** | `cohort` | 7 · race, ethnicity, dialysis rate | Harshini |
| | `model` | 8 · rung 1 NUTS | Harshini |
| | `eval` | 9 · calibration, recovery, fairness | Harshini |
| | `api` | 10 · the tier decision | Rahul |
| | `demo` | 11 · wire UI to API, offline proof | Rahul |
| | `ui` | 12 · the model report screen | Rahul |

**If you fall behind, 1, 2 and 4 are the ones that matter.** Everything else is upside.
Drop rung 1 (8) before you drop any of those — a prior-only model you can explain beats a
NUTS fit you cannot demo.

---

## Round A — paste all six now

`design.py`, `priors.py` and `eha.py` already carry the full medication term structure; the
columns are simply zero. So 1 and 2 are genuinely parallel: the simulator plants medication
coefficients that activate by themselves the moment 1 lands.

### 1 · `cohort` — medications

> Read the **Medication** section of `data/README.md` and `docs/SPEC.md` §5.3. Implement
> `leeward/cohort/medications.py`.
>
> For each veteran, take the active RxNorm codes, map them to VA drug classes with
> `data/reference/va_drug_class_members.parquet`, join
> `data/reference/med_climate_risk.csv`, and derive `med_thermoreg_score` (Σ weight over
> heat-mechanism classes), `acb_score` (Σ acb, the 0–3 ACB scale), `med_combo_raas_diuretic`
> (an ACE inhibitor or ARB *and* a diuretic — the combination CDC names explicitly),
> `med_renal_triple`, `med_cold_chain`, `med_controlled` and `med_narrow_ti`.
>
> Add synthetic `mail_order_pharmacy` (Bernoulli 0.80, from VA's published ~80% mail share)
> and `days_supply_remaining` (90-day fill if mail order else 30-day, uniform phase), both
> with `_synthetic` siblings.
>
> **Make no network calls** — the crosswalks are committed precisely so RxNav is not a
> demo-day dependency.
>
> **Write the test first.** On the Synthea FHIR sample: ~77% of patients with active meds have
> `med_thermoreg_score > 0`, ~16% have `med_combo_raas_diuretic`, ~10% `med_controlled`,
> ~9% `med_cold_chain` — each within 10 percentage points.
> Run `make check`; green; commit; push; two lines in `status/cohort.md`; stop.

### 2 · `model` — the simulator  ← longest pole, start first

> Read `docs/SPEC.md` §5.4. Implement `leeward/cohort/simulate.py` and commit `data/truth.json`.
> Build the linear predictor with `leeward/model/design.py` — **do not fork it**, the whole
> point is that the simulator and the model share one feature mapping — then draw Bernoulli
> outcomes for 120 days into `data/outcomes.parquet` via `schema.write`.
>
> **Write the test first.** With all hazards zeroed, the mean daily rate per need is within
> 30% of `sigmoid(alpha_k)`. On `site_down` days, the treatment-gap rate for dialysis patients
> at that facility at least triples. Seed everything.
>
> `truth.json` must include coefficients for the medication interactions already present in
> `design.py` — `theta_heat_x_meds`, `theta_heat_x_acb`, `theta_heat_x_raas_diuretic`,
> `theta_outage_x_cold_chain`, `theta_sitedown_x_controlled`, `theta_mail_x_supply`. The
> medication columns are zero until prompt 1 lands, so those terms contribute nothing today
> and switch on by themselves afterwards. Plant them now or the model can never recover them.
>
> Do not modify `design.py`; another terminal may be reading it.
> Run `make check`; green; commit; push; two lines in `status/model.md`; stop.

### 3 · `eval` — the test suite breaks when you run the real pipeline  ← cheapest win on the board

`make cohort && make score` leaves `make check` red, and neither failure is a product bug.
The suite reads whatever happens to be in `data/*.parquet`, so it silently assumes fixtures.
Anyone who runs the real pipeline and then runs the gate will chase a ghost at 2am.

Two assertions in `tests/test_allocate.py` are simply wrong for real data:

- `test_fixture_actions_match_the_contract` asserts every scored day gets a list. On the real
  120-day Sandy run, **14 days correctly produce nothing**: max heat 81.9 °F (under the 82 °F
  threshold), no smoke, no outage, no flood, peak risk 0.0476 — under the 0.05 self-serve
  floor, so the whole panel is `everyday` tier. A care team doing nothing on a calm June day
  is the right answer.
- `test_group_floor_reserves_each_borough_its_share` fails only on mixed fixture/real state.
  `group_floor` itself is fine: verified on the real cohort, `{"borough": 0.2}` gives each of
  five boroughs exactly 4 of 20 calls.

> Read `tests/test_allocate.py` and `tests/conftest.py` if one exists. Make the test suite
> independent of whatever is sitting in `data/`: have tests build the frames they need, or
> read from a dedicated `tests/fixtures/` directory that `make fixtures` does not touch.
> Then fix the two assertions above — a day where the whole panel is `everyday` tier must be
> allowed to produce zero actions, and assert that explicitly rather than by accident.
> Acceptance: `make cohort && make score && make check` is green, and so is
> `make fixtures && make check`. Run `make check`; green; commit; push; stop.

### 4 · `api` — the FastAPI app

> Read `docs/SPEC.md` §9 and `leeward/api/schemas.py`. Implement `leeward/api/main.py` with
> all eight routes, returning exactly the pydantic models already frozen in `schemas.py`.
>
> **Never run inference inside a request** — every route reads cached parquets through
> `leeward.schema.read`. `POST /actions` calls `leeward.decision.allocate.allocate` with the
> capacity from the body; that is the capacity slider, and it must answer in under 300 ms.
>
> `leeward/outreach/messages.py` is being written in another terminal right now. Do not write
> it. Have `GET /message/{action_id}` import it lazily and return HTTP 503 with a clear body
> if the import fails, so this lane is never blocked on that one.
>
> Acceptance: `bash scripts/smoke_demo.sh` passes — it boots the API with the network blocked
> and asserts the shape of every route. Run `make check`; green; commit; push;
> two lines in `status/api.md`; stop.

### 5 · `demo` — verified messages

> Read `docs/SPEC.md` §8 and the `Message` model in `leeward/api/schemas.py`. Implement
> `leeward/outreach/verify.py` — a deterministic four-word phrase per veteran-day, drawn from
> a 512-word list of common words, seeded so the same veteran-day always yields the same
> phrase — and `leeward/outreach/messages.py` exposing `render(action, veteran) -> Message`.
>
> `tests/test_guardrails.py` already holds the acceptance tests for this module and they are
> skipping right now. **Read them first; they are your specification.** Every message carries
> the VA channel tag, the four-word phrase, "The VA will never ask you to pay, wire money, or
> share bank details", VSAFE 833-388-7233, and "Veterans Crisis Line: dial 988, press 1".
> No URL shorteners, and no phone number outside the allow-list.
>
> Caregiver-addressed messages name the veteran, speak to the caregiver in the second person,
> and never include a diagnosis. Messages to `low_assets` veterans state what is free and
> never suggest a paid option.
>
> Do not touch `leeward/api/main.py`; another terminal owns it this round.
> Run `make check`; green; commit; push; two lines in `status/demo.md`; stop.

### 6 · `ui` — veteran card and message screen

> Read `leeward/api/schemas.py` — `VeteranCard`, `NeedScore`, `MedicationFlags`, `Message`.
> Build `ui/src/screens/VeteranCard.tsx` and `ui/src/screens/Message.tsx`.
>
> **VeteranCard** is the screen the whole pitch lands on: five needs as 10–90% interval bars
> with the epistemic share shaded, drivers as plain-language chips, a medication panel
> (thermoregulatory score, ACB, the CDC-named combination, cold-chain, controlled, days of
> supply, mail order), the planned actions, and a single "why this tier" line.
>
> **Message** renders the outreach text with the five mandatory elements as a *visible
> checklist*, so a judge can see at a glance that a scammer could not reproduce it.
>
> The live API is being built in another terminal. Read the fixture JSON in
> `ui/public/fixtures/` for now and keep every type imported from the frozen schema shapes, so
> the swap to real routes is a one-line base-URL change.
>
> `npm run build` must be clean. Run `make check`; green; commit; push;
> two lines in `status/ui.md`; stop.

---

## Merge 1 — then run this, it is the moment everything changes

```bash
git checkout main
for l in cohort model eval api demo ui; do git merge --no-ff lane/$l && make check || break; done
make cohort && make score && make check
```

`make cohort` **before** `make score`, or the planted medication truth will not match the
cohort that produced the scores.

Real outcomes now exist, the API is live, and the medication columns go non-zero — which
lights up `pharmacist_med_review` and `cold_chain_plan`, neither of which has ever fired.

**Rahul decides the tier question here** (prompt 10), because it changes what `api` builds next.

---

## Round B — paste all six

### 7 · `cohort` — race, ethnicity, and the dialysis rate

> Read `leeward/cohort/build.py` and `data/README.md`. Three fixes, one commit each.
>
> 1. **`race` and `ethnicity` are null for all 10,000**, so the fairness audit cannot stratify
>    by them — but the deck quotes NYC Health's finding that Black New Yorkers die of heat
>    stress at three times the white rate, and §9 of the proposal lists race in the audit.
>    Draw both from a real per-ZIP source (CDC SVI `EP_MINRTY` is already in
>    `data/reference/svi_nyc_tract.parquet`; aggregate tract → MODZCTA) so the distribution is
>    grounded rather than invented, and flag them `_synthetic`. If you conclude no defensible
>    source exists, say so in `status/cohort.md` and leave them null — do not invent a
>    distribution to make a table render.
> 2. **Only 7 veterans are on dialysis (0.07%)**, because emPOWER's per-ZIP facility-dialysis
>    count is small. The Sandy dialysis story is the emotional core of the pitch and it
>    currently rests on seven people. Raise the rate to the VA's own ESRD prevalence among
>    enrolled veterans, cite it in `docs/sources.md`, and state the change in the PR.
> 3. **`med_cold_chain` and `med_controlled` are zero for everyone** because the medication
>    layer has not landed. Leave them; the next task fills them.
>
> Write the test first in each case. Run `make check`; green; commit; push; stop.

### 8 · `model` — rung 1, pooled NUTS

> Read `docs/SPEC.md` §6. Implement `leeward/model/hazard.py` and `leeward/model/fit.py` for
> **rung 1 only**: binomial-cell likelihood, five needs, `alpha`, health betas, and the 4-lag
> heat curve. **No ICAR, no latent dose, no interactions** — leave TODOs and flags.
>
> Write `data/posterior.nc` as ArviZ InferenceData with variable names matching `priors.py`,
> and make `score.py` emit `model_rung = 1`.
>
> Fit first on a 200-veteran, 30-day toy cohort; it must complete in under 60 seconds with
> zero post-warmup divergences. Then report r-hat and divergences for the full fit and write
> both into `report/fit.json`.
>
> Do not change `design.py` without asking. Run `make check`; green; commit; push;
> two lines in `status/model.md`; stop.

### 9 · `eval` — calibration, recovery, fairness

> Read `docs/SPEC.md` §11. Implement `leeward/eval/calibration.py` (reliability per need on a
> held-out window, ECE), `leeward/eval/recovery.py` (true coefficients from `truth.json` vs
> 90% posterior intervals, with coverage), and `leeward/eval/fairness.py` (ECE and false
> negative rate by borough, HVI band, evacuation zone, income band, caregiver status,
> medication burden and race/ethnicity).
>
> `fairness.py` must mark any group whose relative FNR gap exceeds 20% as `flagged`.
> **A failing audit is displayed, never suppressed** — `test_guardrails.py` asserts that this
> module contains no swallowed exceptions, and that assertion turns on the moment you create
> the file. Assemble everything into `report/report.json` matching `ReportResponse`.
> Run `make check`; green; commit; push; two lines in `status/eval.md`; stop.

### 10 · `api` — make the tier column mean what the deck says

On the Sandy SiteDown day, **29 of 40 call slots went to `self_serve` veterans while 33
`act_now` veterans got no call at all.** Across the run, 98% of expensive actions
(10,198 of 10,436) went to `self_serve`, and 70% of `act_now` veterans received only an
automated text.

`allocate.py` is not buggy — it implements SPEC §7.5 faithfully. §7.5 only uses tier to gate
`check_in_call` and to grant act_now three slots; it never says act_now gets *priority* for a
scarce bucket. Allocation is pure greedy-by-EHA, so a self-serve veteran with broad moderate
risk outranks an act-now veteran with one sharp high risk. **The spec is under-specified and
the proposal promises something else** — §6's tier table says Act now → "Call today, VA care
team" and Self-serve → "Verified text, automated".

Two coherent resolutions. Rahul picks:

- **(A) Tier gates the action class.** Expensive/clinical actions are offered act_now first,
  then find_out, then self_serve; EHA ranks *within* a tier. The tier column then means what
  the deck says. Costs a little total EHA.
- **(B) Keep pure EHA maximisation** and change the deck: drop the tier→action mapping and say
  "we spend the call where it averts most harm, not where a label says to." Defensible, maybe
  even stronger — but then the tier badge must stop implying an action.

> Read `leeward/decision/allocate.py`, `docs/SPEC.md` §7.5 and `docs/proposal.md` §6.
> Implement resolution **<A or B>**. If A: add a tier priority band to the allocation order so
> scarce buckets fill act_now before find_out before self_serve, EHA-ranked within each band,
> and update SPEC §7.5 to say so. If B: update SPEC §7.5 and proposal §6 to state that tiers
> describe urgency, not entitlement, and stop the UI implying an action from a tier.
> Either way add a test asserting the chosen behaviour, and one asserting total EHA does not
> fall by more than 15% versus today. Run `make check`; green; commit; push; stop.

### 11 · `demo` — wire the UI to the live API, then prove it offline

> `leeward/api/main.py` now exists, so the UI can stop reading `ui/public/fixtures/`.
> Point `ui/src/lib/api.ts` at the live routes, keep the fixture path as an automatic
> fallback when the API is unreachable, and make `make demo` boot API and UI together.
>
> Then prove it offline: run `bash scripts/clean_clone_test.sh` and
> `bash scripts/smoke_demo.sh`, and fix whatever they catch. Both must pass **with the wifi
> physically off**, not by trusting that they would.
>
> Acceptance: `make demo` reaches a rendered care-team list in under 60 seconds from a clean
> clone with no network and no `.env`. Write the wall-clock boot time into `status/demo.md`.
> Run `make check`; green; commit; push; stop.

### 12 · `ui` — the model report screen

> Read `ReportResponse`, `RecoveryRow`, `CalibrationBin`, `FairnessRow`, `AblationRow` and
> `DecisionQualityRow` in `leeward/api/schemas.py`. Build `ui/src/screens/Report.tsx`:
> the recovery dot-and-whisker, reliability curves per need, the harm-averted bar chart
> against the three baselines, the ablation table, and the fairness table.
>
> **The fairness table renders whether or not it passes, and flagged rows are visually
> marked.** A failing audit is displayed, never suppressed — that is a project rule, and
> `tests/test_guardrails.py` enforces the backend half of it.
>
> `npm run build` clean. Run `make check`; green; commit; push;
> two lines in `status/ui.md`; stop.

---

## Merge 2 → Round C is rehearsal only

```bash
git checkout main
for l in cohort model eval api demo ui; do git merge --no-ff lane/$l && make check || break; done
make cohort && make score && make report && make check
bash scripts/clean_clone_test.sh && bash scripts/smoke_demo.sh
```

After this, **no new features.** Round C is three timed run-throughs, the backup video, and
the cut list in `docs/BUILD_PLAN.md` §6. See §8 of that file for the demo-engineering
checklist — offline by construction, seeded, pre-warmed, no live typing.

---

## Prompts for the two of you, not for an agent

**Every 90 minutes, whoever is free:**

> Run `make status` and `make check`. If check is red, name the failing test and fix only that.
> Do not fix unrelated failures. Then merge every green lane branch into main and push.

**When a lane finishes a task and you want the next one:** paste the next prompt from this
file. Do not ask an agent to "continue" — a fresh, fully-specified prompt beats a vague
follow-up, especially when nobody is reading the output.

**When something looks wrong but you do not want to read code:**

> `make status` shows X but I expected Y. Find the cause, write a failing test that reproduces
> it, then fix it. Show me the test, not the fix.

That last one is the highest-value habit on this project. A failing test you can read in ten
seconds is worth more than a diff you are not going to read at all.

---

## The five things that are never cut

Walter's card · the capacity slider · the calibration plot · the SiteDown term ·
the verified-message screen.

Everything else is in the cut list in `docs/BUILD_PLAN.md` §6, in order. When a checkpoint
slips by more than 45 minutes, take the next item off it without discussion.
---

## Appendix — Wave 1, merged

Kept for reference. All six landed; `make check` green at 255 passed. Running the real
pipeline afterwards surfaced prompts 3, 7 and 10 above.

## Wave 1 — all six can run at once, right now

Everything in Wave 1 depends only on `data/reference/` and the fixtures, both of which
already exist. Start all six simultaneously.

### `cohort` — the real 10,000-veteran panel

> Read `CLAUDE.md`, then `data/README.md`, then `docs/SPEC.md` §5. Implement
> `leeward/cohort/build.py` producing `data/cohort.parquet` for 10,000 synthetic NYC veterans.
>
> Draw every neighbourhood rate from `data/reference/` — never invent one that exists there:
> `acs_veterans_by_zcta.parquet` for where veterans live by age band (this sets the ZIP
> sampling weights), `places_zcta_nyc.parquet` for mobility, social support, utility-shutoff
> risk, transport barriers and chronic-disease prevalence, `empower_ny_zip.parquet` for
> powered equipment, `hvi_by_zcta.parquet`, `evac_zone_by_modzcta.parquet` and
> `stormwater_by_modzcta.parquet` for hazard exposure, and
> `va_facilities_nyc_hazard.parquet` for facility assignment.
>
> Use `leeward.schema` for the column contract and call `schema.write(df, "cohort")` — do not
> call `write_parquet` directly. Every synthetic column needs its `_synthetic` sibling.
> Seed everything; `--seed` defaults to 0.
>
> **Write `tests/test_cohort.py` first.** It must assert: 10,000 rows; every `modzcta` is one
> of the real 178; the realised rate of each PLACES-derived field is within 20% of the
> ZIP-weighted source rate; ZIP frequencies correlate above 0.7 with ACS veteran counts.
>
> Do not touch `leeward/schema.py`, and do not implement the simulator or the model.
> Run `make check`; it must be green. Commit, push, write two lines to `status/cohort.md`, stop.

### `model` — rung 0, prior-only, no MCMC

> Read `docs/SPEC.md` §6, especially §6.0 (the ladder). Implement **rung 0 only**:
> `leeward/model/priors.py`, `leeward/model/design.py` and `leeward/model/score_prior.py`.
>
> `design.py` builds the linear predictor from a cohort frame and a hazards frame. It is
> shared by the simulator and the model later, so keep it pure and free of I/O.
> `score_prior.py` draws 400 coefficient vectors from the priors in `priors.py`, computes
> risk by matrix multiply, and writes `data/scores.parquet` via `schema.write`, with
> `model_rung = 0`.
>
> There is **no MCMC in this task**. This is the demo's floor: from the moment it lands, every
> screen has defensible numbers.
>
> **Write the test first.** Assert: all five needs present for every veteran-day; every
> probability in (0, 1); `p_lo80 <= p_mean <= p_hi80` everywhere; `p_epistemic_share` in [0, 1];
> scoring 10,000 veterans × 7 days takes under 5 seconds.
>
> Do not touch `leeward/schema.py`. Run `make check`; green; commit; push;
> two lines in `status/model.md`; stop. I will ask for rung 1 separately.

### `api` — the decision layer

> Read `docs/SPEC.md` §7 and `leeward/api/schemas.py`. Implement `leeward/decision/severity.py`
> (the `w_k` weights as editable YAML), `leeward/decision/tau.py` (the `τ[a,k]` table),
> `leeward/decision/eha.py` and `leeward/decision/allocate.py`.
>
> `allocate.py` must expose `allocate(scores, cohort, capacity, group_floor=None) -> pl.DataFrame`
> matching the `actions` contract in `leeward.schema`, and `total_eha(actions) -> float`.
> Greedy by EHA per unit cost, capped per capacity bucket, at most one action per veteran
> unless `tier == "act_now"` (then at most three). `rank` is 1..n with no gaps, ordered by
> descending EHA.
>
> `tests/test_guardrails.py` already contains the acceptance tests for this module and they
> are currently skipping. Read them first — they are your specification. In particular
> `test_more_capacity_never_averts_less_harm` requires that raising any capacity never lowers
> total EHA.
>
> Also write `tests/test_allocate.py` with a five-veteran fixture where the optimum is
> hand-checkable. Work against `data/*.parquet` fixtures, not real scores.
> Run `make check`; green; commit; push; two lines in `status/api.md`; stop.

### `ui` — the map and the care-team list

> Read `leeward/api/schemas.py` for the response shapes, and `docs/SPEC.md` §10.
> Scaffold a Vite + React 18 + TypeScript app in `ui/`, then build two screens:
>
> 1. **Map** — deck.gl `GeoJsonLayer` over `data/reference/nyc_modzcta.geojson` (178 real NYC
>    ZIP polygons), filled by expected need count, plus an `IconLayer` of VA facilities from
>    `data/reference/va_facilities_nyc_hazard.parquet` that turns red when a facility is down.
> 2. **CareTeam** — POST `/actions` with `{date, capacity}`, render a ranked table (rank, name,
>    tier badge, top driver, EHA, owner), a `CapacitySlider` (10–100) that refetches on release,
>    and a harm-averted counter that animates between values.
>
> Until the API exists, read the fixture JSON directly — but type everything against
> `leeward/api/schemas.py`, because those shapes are frozen.
>
> No UI kit beyond what you add to `package.json`. `npm run build` must succeed.
> Acceptance: `npm run build` clean, both screens render from fixtures, slider round-trip under
> 300 ms. Run `make check`; green; commit; push; two lines in `status/ui.md`; stop.

### `demo` — scenarios and the hazard assembler

> Read `data/README.md` and `docs/SPEC.md` §3.2 and §5.5. Implement
> `leeward/ingest/hazards.py`: given a scenario YAML and the tables in `data/reference/`,
> emit `data/hazards.parquet` (one row per modzcta × day, 120 days) and
> `data/site_status.parquet` (facility × day), both via `schema.write`.
>
> Write three scenarios in `scenarios/`: `sandy_then_heat.yaml`, `ida_flash_flood.yaml`,
> `smoke_2023.yaml`. Heat, surge, outage and flood come from the scenario. `evac_zone_min`,
> `stormwater_flooded_frac` and `hvi` are static per-ZIP joins.
>
> Two things must be **replayed, not simulated**: the smoke scenario reads real PM2.5 from
> `airnow_pm25_nyc_smoke2023.parquet` by nearest monitor, and the Sandy scenario marks
> **station 630 down**, because station 630 is the Manhattan VA and it sits in evacuation
> zone 1 — check `va_facilities_nyc_hazard.parquet`.
>
> **Write the test first.** Assert: every modzcta appears on every day; no nulls; on the smoke
> scenario's peak day the citywide max PM2.5 exceeds 190 µg/m³ (the real 7 June 2023 value was
> 203.5); on Sandy day 63 `site_status` has station 630 down.
>
> Make no network calls. Run `make check`; green; commit; push; two lines in `status/demo.md`; stop.

### `eval` — the Impact bar chart

> Read `docs/SPEC.md` §11. Implement `leeward/eval/decision_quality.py`: for each day in a
> held-out window and K in {20, 40, 80}, select actions with `leeward.decision.allocate` and
> with three baselines — rank by age, rank by `n_chronic`, and random seeded at 0 — then
> compute harm averted as `Σ w_k · τ[a,k] · y_true[i,k,t]` over the selected veterans.
> Emit a tidy CSV and a Plotly bar chart into `report/`.
>
> If `leeward.decision.allocate` does not exist yet, code against the interface in
> `tests/test_guardrails.py::test_more_capacity_never_averts_less_harm` and leave it failing
> only on the import — do not stub the allocator.
>
> **Write the test first**, on a tiny fixture where Leeward must beat random.
> Run `make check`; green; commit; push; two lines in `status/eval.md`; stop.

---

# Leeward — the two-person build plan

*Health in Climate AI Hackathon, Cornell Tech, 19–20 September 2026 · 2 builders · 6 Claude Code terminals · one working day*

This replaces §12 of `docs/SPEC.md`, which was written for four people over twenty-six hours.
Everything below assumes **two people, starting now, with nothing built.**

---

## 0. What is already done, and what it buys you

Before you write a line of code, `data/reference/` already holds every public source the
build needs, fetched, joined, verified and committed. See `data/README.md` for the catalog
and `docs/sources.md` for every URL.

| Was going to cost you | Status |
| --- | --- |
| Finding live endpoints for 10 agencies | **Done.** `scripts/fetch_sources.py`, 18 fetchers, all keyless |
| emPOWER ArcGIS paging | **Done.** 1,702 NY ZIPs, `empower_ny_zip.parquet` |
| HVI → ZIP crosswalk via NTA | **Not needed.** NYC now publishes HVI per ZCTA20 directly |
| Evacuation-zone polygons → ZIP | **Done.** Area-weighted, `evac_zone_by_modzcta.parquet` |
| FloodNet API access request form | **Not needed.** Same data is open on NYC Open Data |
| AirNow API key | **Not needed.** `files.airnowtech.org` is keyless |
| VA Facilities API key | **Not needed.** VHA ArcGIS mirror is keyless, 84 NY sites |
| Census API key | **Worked around.** `api.census.gov` now 302s to `missing_key.html`; the ACS Summary File path is keyless |
| FEMA NRI static zip (now 301s to a landing page) | **Fixed.** Live FEMA FeatureServer, 2,324 NYC tracts |
| Building a heat-sensitive drug list by hand | **Done.** 52-class crosswalk from CDC's clinician guidance, plus 4,222 RxNorm→VA-class mappings from RxNav |

**Net: roughly four hours of Track-A work is already in the repo.** Hour 0 is not an ingest
hour any more. Spend it on the spine.

Four findings from that work change the pitch, not just the schedule — read §7 before you
write the deck.

---

## 1. The two rules that make two people work like four

**Rule 1 — Spine before substance.** By T+2h there is a running end-to-end demo with fake
numbers: forecast → map → ranked list → veteran card → message. Every hour after that
replaces one fake with one real thing. You are never in a state where there is nothing to
show. If the model never converges, you still demo.

**Rule 2 — Contracts before parallelism.** Six terminals can only run in parallel if they
agree on the shape of the data. The first 45 minutes produce `schema.py`, `api/schemas.py`
and a **fixture generator** that emits correctly-shaped fake parquets. After that, every
lane develops against fixtures and nobody blocks anybody.

---

## 2. Who owns what

Rahul runs the live demo, so Rahul owns everything the judges will see on screen.

| | **Rahul — "Demo"** | **Partner — "Truth"** |
| --- | --- | --- |
| Owns | API, decision layer, outreach, UI, scenarios, deck, rehearsal | Cohort, simulator, model, scoring, evaluation, fairness |
| Ships | A demo that never breaks | Numbers that survive a judge asking "how do you know?" |
| Contract files owned | `leeward/api/schemas.py` | `leeward/schema.py`, `leeward/model/design.py` |
| Worktrees | `wt/api`, `wt/ui`, `wt/demo` | `wt/cohort`, `wt/model`, `wt/eval` |

**Nobody edits a contract file they do not own.** If you need a column, ask in chat; the
owner adds it and pushes within five minutes. This single rule prevents the merge conflicts
that kill two-person hackathon teams.

### Terminal discipline

Three terminals per person, but **only two are ever actively steered**. The third runs long
jobs (`make fit`, `make report`, the 4 GB download) and is checked, not watched.

```bash
# one time, at the top of the hackathon
git worktree add ../lw-api   -b lane/api
git worktree add ../lw-ui    -b lane/ui
git worktree add ../lw-demo  -b lane/demo
git worktree add ../lw-cohort -b lane/cohort
git worktree add ../lw-model  -b lane/model
git worktree add ../lw-eval   -b lane/eval
```

Each terminal: `cd ../lw-<lane> && claude`. Separate worktrees mean two agents editing the
same repo never fight over the working tree.

### Merge protocol

```bash
# in a lane, when green
pytest -q && git add -A && git commit -m "lane/<x>: <what>" && git push -u origin lane/<x>
# on main, by the lane owner, no PR ceremony
git checkout main && git pull && git merge --no-ff lane/<x> && pytest -q && git push
```

Merge to `main` **every 60–90 minutes minimum**, even if incomplete. A lane that has not
merged in three hours is a lane that is about to have a four-hour merge.

---

## 3. The model ladder — the single biggest risk, defused

A hierarchical NumPyro model with ICAR spatial effects, a latent exposure dose, distributed
lags and five correlated needs is a two-day research task, not a one-day build task. So it
is built as a **ladder**. Every rung writes the same `data/posterior.nc` contract, so
nothing downstream changes when you climb.

| Rung | What it is | Time | Ship it if |
| --- | --- | --- | --- |
| **0 — Prior-only** | No MCMC. Coefficients drawn from `priors.py` means, risk computed by matrix multiply. Intervals from prior draws. | 30 min | Always. Build this first. It is the demo's floor. |
| **1 — Pooled NUTS** | NumPyro on binomial cells. Five needs, main effects, the 4-day heat lag. No ICAR, no latent dose, no interactions. | 2 h | This is the realistic target |
| **2 — Interactions + SiteDown** | Adds the six interaction terms and the facility-closure term. This is where the story lives. | 2 h | Strong finish |
| **3 — ICAR + latent dose** | Spatial pooling and the burn-pit measurement model. | stretch | Only if rung 2 is merged by T+12h |

**Rung 0 is not a fallback, it is the first deliverable.** Write it in the first two hours.
It guarantees that at every moment from T+2h onward, `make demo` produces a working,
defensible demo. Everything above rung 0 is upside.

Say the rung you reached on stage. "We fit the pooled hierarchical model and the
interactions; the spatial prior is in the repo behind a flag and did not converge in time"
is a strong answer. A model that silently did not fit is not.

---

## 4. The schedule

Clock times assume a 16:00 Saturday start and a 15:00 Sunday submission. `T+` offsets are
the real contract — shift them if you start later.

### T+0 → T+0:45 · The contract freeze — **both people, one screen, no agents**

Do this together, out loud, in one terminal. It is the highest-leverage 45 minutes of the
event and it is the one thing you should not delegate.

1. `leeward/schema.py` — polars schemas for `cohort`, `hazards`, `outcomes`, `scores`,
   `actions`, `outcome_log`. Copy from SPEC §3, change `zip` → `modzcta` throughout.
2. `leeward/api/schemas.py` — pydantic v2 request/response models for the eight routes.
3. `scripts/make_fixtures.py` — emits fake-but-correctly-shaped parquets into `data/`.
   500 veterans, 30 days, random risks. **This is what unblocks all six lanes.**
4. `make fixtures && pytest -q tests/test_schema.py` green.
5. Commit to `main`, push, both people pull.

> **Prompt (run once, together):**
> Read `CLAUDE.md` and `docs/SPEC.md` §3. Write `leeward/schema.py` with polars schemas for
> the six contract tables, using `modzcta` (not `zip`) as the geography key, and
> `leeward/api/schemas.py` with pydantic v2 models for the routes in §9. Then write
> `scripts/make_fixtures.py` that generates a 500-veteran, 30-day fixture set satisfying
> every schema, seeded at 0. Write `tests/test_schema.py` first: it must load each fixture
> and assert the schema. Do not implement any model, API route, or UI.

### T+0:45 → T+2:30 · Spine

| Lane | Task |
| --- | --- |
| `api` | FastAPI app, all 8 routes, reading fixtures from disk. No logic, correct shapes. |
| `ui` | Vite + React + deck.gl skeleton. Map of 178 MODZCTAs from `data/reference/nyc_modzcta.geojson`, coloured by a fixture column. |
| `demo` | `Makefile` (`fixtures data cohort fit score demo report test`), `pyproject.toml`, `.env.example`, clean-clone test script. |
| `cohort` | `cohort/build.py`: read `data/reference/*`, emit 10,000 veterans with real ZIP-level priors. **No Synthea yet** — draw demographics parametrically. |
| `model` | Rung 0: `model/priors.py`, `model/design.py`, `model/score_prior.py`. |
| `eval` | `decision/severity.py`, `decision/tau.py`, `decision/eha.py`, `decision/allocate.py` against fixtures. |

**Checkpoint T+2:30 — the first `make demo`.** If the browser shows a map, a ranked list and
a veteran card driven by rung-0 scores, you are on schedule. If not, cut UI polish until it does.

### T+2:30 → T+6:00 · Make it real

| Lane | Task |
| --- | --- |
| `cohort` | Swap parametric demographics for Synthea. Use `data/raw/synthea_sample_fhir.zip` for the FHIR code path and its tests; use the VA 100k CSV release for the 10,000-veteran cohort if the download has landed, otherwise say so and move on. |
| `cohort` | `cohort/augment.py` using the **real** ZIP priors (see §5) and `cohort/missingness.py`. |
| `cohort` | `cohort/medications.py` — the medication layer (§5b). Cheap and high-yield: the RxNorm codes are already in the record and the crosswalks are already committed. |
| `model` | `cohort/simulate.py` sharing `design.py`, `truth.json`, 120 days. Then rung 1: `model/hazard.py`, `model/fit.py`. |
| `api` | Real EHA + greedy allocation + tiers. `outreach/messages.py` + `verify.py` with all five mandatory elements. |
| `ui` | Care-team list, capacity slider, veteran card with interval bars, message screen. |
| `demo` | `scenarios/*.yaml`. Three scenarios, two of them replays of real data (§7). Offline snapshot pack. |
| `eval` | `eval/calibration.py`, `eval/recovery.py`, `eval/decision_quality.py` on fixtures. |

**Checkpoint T+6:00 — `make cohort && make score && make demo` end to end on real data.**

### T+6:00 → T+9:30 · The proof

| Lane | Task |
| --- | --- |
| `model` | Rung 2: interactions + SiteDown. Start `make fit` in the third terminal and leave it. |
| `eval` | Parameter recovery, calibration on days 91–120, PPC, ablations, **fairness audit**. |
| `api` | `GET /report`, `GET /export`, `POST /log`. Caregiver routing in `allocate.py`. |
| `ui` | Model report screen: recovery dot-whisker, reliability curve, harm-averted bars, fairness table. |
| `demo` | Deck v1, eight slides. Walter's card wired to a real veteran id. First full run-through. |

**Checkpoint T+9:30 — a judge could ask "how do you know it works?" and you open a screen.**

### T+9:30 → T+11:00 · Freeze and first rehearsal

Both people, one screen. Full run-through, timed. Write down every stumble. Fix only
stumbles. **Record the backup video tonight**, not tomorrow.

### T+11:00 → T+16:30 · Sleep

Leave `make report` and any rung-3 fit running. Yes, really sleep. A rehearsed demo beats an
extra feature, every time.

### T+16:30 → T+21:00 · Polish, not features

| Lane | Task |
| --- | --- |
| `demo` | Clean-clone test in `/tmp`. Time `make demo`. Must be under 60 s. Rehearse ×3. |
| `ui` | Only what the rehearsal exposed. No new screens. |
| `api` | Bug bash. Every route returns in under 300 ms from cache. |
| `eval` | Final numbers into the deck. Fairness table rendered even if it fails. |
| `model` | Freeze. Merge whatever rung is green. Write down the rung. |

### T+21:00 → T+23:00 · Submission

Two-pager, 90-second video, repo public, eight slides, final rehearsal, `docs/sources.md`
reconciled against every number on screen.

---

## 5. Use the real priors — this is free credibility

The SPEC's original `augment.py` table invented rates (`mobility_impaired = 0.18 for 65+`,
`caregiver: none 0.35`, `home_ac` by HVI band). **Do not ship invented rates.** CDC PLACES
now publishes all of them per ZCTA, and they are already in
`data/reference/places_zcta_nyc.parquet`.

Measured across NYC's 178 MODZCTAs, grouped by Heat Vulnerability Index band:

| HVI | ZIPs | Utility shutoff threat | Lacks emotional support | Mobility difficulty | Lacks transport | COPD | Independent-living difficulty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (lowest) | 34 | 5.3% | 24.8% | 9.3% | 6.0% | 3.9% | 5.5% |
| 2 | 36 | 6.3% | 26.6% | 10.6% | 7.3% | 4.3% | 6.2% |
| 3 | 33 | 8.9% | 30.8% | 13.2% | 10.1% | 5.3% | 7.7% |
| 4 | 37 | 11.9% | 33.2% | 15.2% | 12.8% | 5.8% | 9.2% |
| 5 (highest) | 37 | **17.2%** | **35.0%** | **18.8%** | **16.6%** | **6.7%** | **11.4%** |

Every gradient is monotone in HVI, from an entirely independent source. Map the augment
fields straight onto these columns:

| cohort column | PLACES column | why |
| --- | --- | --- |
| `low_assets` / cannot run the AC | `shututility_crudeprev` | NYC Health's finding is that people died with an AC they could not afford to run. This is that, per ZIP: 5.3% → 17.2% across the heat gradient. |
| `caregiver == none` | `emotionspt_crudeprev`, `loneliness_crudeprev` | "Lacks social and emotional support" is the measured version of the thing clinicians look for first |
| `mobility_impaired` | `mobility_crudeprev` | |
| transport barrier for rides | `lacktrpt_crudeprev` | |
| `copd`, `asthma`, `active_cancer_tx`, `depression` base rates | `copd_`, `casthma_`, `cancer_`, `depression_crudeprev` | |
| `powered_equipment` | `empower_ny_zip.parquet` | 36,146 electricity-dependent Medicare beneficiaries in NYC; 3,165 on oxygen; 1,948 facility ESRD dialysis |

Keep `home_ac`, `floor`, `burn_pit_years` and `evac_zone` as they are — those genuinely are
assumptions, and `floor` in particular has no public per-person source. Flag them
`_synthetic` and say so. **The line that wins this is: every neighbourhood-level rate in the
cohort is real and cited; only the people are synthetic.**

---

## 5b. The medication layer — the cheapest large gain in the build

Synthea puts an RxNorm code on every prescription. The first draft of the spec collapsed all
of that into one boolean, `heat_sensitive_meds`. Two committed files turn it into a covariate
block, and the work is a couple of joins:

- `data/reference/va_drug_class_members.parquet` — 4,222 RxNorm codes → **the VA's own drug
  classes**, from RxNav. Offline, so no API call at demo time.
- `data/reference/med_climate_risk.csv` — 52 VA classes → mechanism, hazard, weight, ACB
  score, controlled/cold-chain/narrow-TI flags. Curated from CDC's clinician guidance.
  A pharmacist can edit it; nobody has to touch code.

Derive six fields and you have the whole layer:

```
med_thermoreg_score      Σ weight over heat-mechanism classes
acb_score                Σ acb, 0-3 per drug; >=3 is the clinical threshold
med_combo_raas_diuretic  (ACE-i or ARB) and a diuretic   <- CDC names this pair explicitly
med_cold_chain           insulin and friends             -> outage interaction
med_controlled           opioid / benzo / stimulant      -> retail refill route EXCLUDES these
days_supply_remaining    synthetic, 30-day window / 90-day mail
```

Measured on the Synthea FHIR sample, so you know what to expect: **77%** of patients on
active meds carry at least one heat-impairing drug, **16%** carry the CDC-named pair, 10%
a controlled substance, 9% cold-chain, 5% at ACB ≥ 3. The five most-prescribed drugs are
insulin, hydrochlorothiazide, lisinopril, metformin and amlodipine — so these terms fire on
the ordinary patient, not an exotic one.

**The safety line matters and belongs in the demo script:** Leeward flags a medication for
pharmacist review. It never changes a dose. CDC's guidance already tells clinicians to review
medication lists before hot weather; Leeward's contribution is naming which forty patients to
review before Thursday. `pharmacist_slot` is a scarce capacity unit in `allocate.py` for
exactly that reason — it is a real person's afternoon.

---

## 6. Cut list, re-ordered for one day

Cut in this order, without discussion, the moment a checkpoint slips:

1. SBC (simulation-based calibration)
2. Rung 3 — ICAR spatial prior and latent burn-pit dose
3. Prior-scale slider (the three cached posteriors)
4. Partner-export CSV
5. Outcome-log write-back into scoring
6. Ida scenario — keep Sandy-then-heat and the smoke replay
7. deck.gl → static Plotly choropleth
8. Medication interaction *terms* in the model → keep the medication **drivers and actions**,
   which are rule-based and need no posterior
9. Rung 2 interactions → ship rung 1

**Never cut:** Walter's card · the capacity slider · the calibration plot · the SiteDown
term · the verified-message screen. Those five are the demo.

---

## 7. Four things the grounding work changed — put these in the deck

**1. The Manhattan VA is in Evacuation Zone 1. That is a join, not a claim.**

`data/reference/va_facilities_nyc_hazard.parquet` joins the VHA facility registry to NYC's
hurricane evacuation zones:

| Station | Facility | Evac zone | Site-dependent services |
| --- | --- | --- | --- |
| **630** | **Margaret Cochran Corbin VA Campus (Manhattan)** | **1** | dialysis, infusion, OTP |
| 630GB | Staten Island Community VA Clinic | 2 | |
| 630A4 | Brooklyn VA Medical Center | 4 | dialysis |
| 630A5 | St. Albans VA Medical Center | 6 | |
| 526 | James J. Peters VAMC (Bronx) | 0 | dialysis |

Station 630 is the campus that evacuated on 28 October 2012 as Sandy approached; its opioid
treatment program stayed closed for five months and about 100 veterans needed emergency
guest-dosing across NYC. It sits, today, in the first zone the city orders to evacuate.
**That is the entire argument for the SiteDown term, and you can run the join on stage.**

**2. The smoke scenario is a replay, not a script.**

`airnow_pm25_nyc_smoke2023.parquet` holds real EPA AirNow monitor data for 5–11 June 2023:

| Date | Peak PM2.5 (µg/m³) | Peak AQI |
| --- | --- | --- |
| 6 Jun | 101.0 | 175 |
| **7 Jun** | **203.5** (Queens monitor) | **254** |
| 8 Jun | 106.9 | 178 |
| 9 Jun | 14.9 | 57 |

Thirteen to 203 and back in four days. Do not simulate this — replay it. A judge can check
the number.

**3. The 82 °F hinge is NYC Health's own finding, not a modelling choice.**

The 2026 NYC Heat-Related Mortality Report attributes the rise in heat-exacerbated deaths
mainly to more **"non-extreme hot days," 82 °F to below the 95 °F extreme-heat threshold**.
The model's heat hinge sits at exactly that number for exactly that reason. Say so — it
turns a hyperparameter into a citation. The same report gives you ~500 premature deaths per
warm season, ~490 heat-exacerbated per year (2014–2023), 7 heat-stress deaths per year
(2016–2025), 19 of them in the June 2025 heat wave alone, and Black New Yorkers dying of
heat stress at three times the rate of white New Yorkers — which is why the fairness audit
is a screen and not a footnote.

---

**4. Four in five VA prescriptions arrive by mail — so a flood stops the pharmacy.**

VA delivers roughly **80% of outpatient prescriptions** through its Consolidated Mail
Outpatient Pharmacy network: about **518,000 prescriptions a day**, reaching **330,000
veterans**. No civilian health system has that concentration, and it means a flooded ZIP is a
medication-supply event, not only a clinic event. It has already failed once for a non-climate
reason — the 2020 USPS slowdown drew a bipartisan congressional letter about delayed veteran
prescriptions.

The sharpest edge is a rule, not a model: **the VA emergency retail refill benefit excludes
controlled substances.** A veteran can walk into any pharmacy with a VA bottle and get a
10-day supply — unless they are on an opioid, a benzodiazepine, a stimulant, or methadone
through an OTP. Those veterans are exactly the ones the workaround does not cover, and exactly
the ones Sandy stranded: about 100 needed emergency guest-dosing when the Manhattan VA's
opioid treatment program closed for five months. Leeward surfaces them first, five days out,
because for them the fallback does not exist.

## 8. Demo engineering — Rahul's checklist

The demo is a system with its own failure modes. Treat it like one.

**Offline by construction.** `make demo` must never touch the network. Everything reads
`data/reference/` and cached parquets. Test it with Wi-Fi physically off, not by trusting
that it would work.

**Determinism.** Seed everything: cohort generation, missingness, the random baseline in
`decision_quality.py`, the verification-phrase generator. The same click must produce the
same number in the rehearsal and on stage.

**Pre-warm.** Boot the API and UI ten minutes before you present, click through every screen
once, leave the tabs open. Cold JIT on the first deck.gl render is a five-second silence at
the worst possible moment.

**One slider, one moment.** The capacity slider from 40 → 20 → 80 with the harm-averted
counter moving is the single most persuasive three seconds you have. Rehearse it until the
number lands while you are still talking.

**No live typing.** Every veteran id, ZIP and date in the demo path is a click or a
preloaded URL. Typing on stage is how demos die.

**Backup video Saturday night.** Ninety seconds, screen-recorded, narrated. If the laptop
fails you still present. Record it at T+9:30, not Sunday morning.

**Two browser windows.** One on the care-team list, one on the model report. Alt-tab beats
navigating.

**Know your rung.** Have the one-sentence honest answer ready: which ladder rung fit, what
the r-hat was, what did not converge.

---

## 9. Claude Code prompts, by lane

Paste these as first messages. One task per prompt; state the acceptance test in the prompt.

**`cohort` — build the cohort on real priors**
> Read `CLAUDE.md`, `docs/SPEC.md` §5 and `data/README.md`. Implement `leeward/cohort/build.py`
> and `leeward/cohort/augment.py`. Draw every neighbourhood rate from `data/reference/`:
> `places_zcta_nyc.parquet` for mobility, emotional support, utility shutoff, transport and
> chronic-disease prevalence; `empower_ny_zip.parquet` for powered equipment; `hvi_by_zcta.parquet`
> for the heat band; `evac_zone_by_modzcta.parquet` for surge exposure. Do not invent a rate that
> exists in those files. Write `tests/test_cohort.py` FIRST: assert that the cohort's realised
> rate for each augmented field is within 20% of the ZIP-weighted source rate, and that every
> augmented column has a `_synthetic` sibling set True. Do not touch `leeward/schema.py`.

**`cohort` — the medication layer**
> Read `data/README.md` (the Medication section) and `docs/SPEC.md` §5.3. Implement
> `leeward/cohort/medications.py`: for each veteran, take the active `MedicationRequest`
> RxNorm codes, map them to VA drug classes with `data/reference/va_drug_class_members.parquet`,
> join `data/reference/med_climate_risk.csv`, and derive `med_thermoreg_score`, `acb_score`,
> `med_combo_raas_diuretic`, `med_renal_triple`, `med_cold_chain`, `med_controlled` and
> `med_narrow_ti`. Add synthetic `mail_order_pharmacy` (Bernoulli 0.80) and
> `days_supply_remaining` (90-day fill if mail order else 30-day, uniform phase), both with
> `_synthetic` siblings. Make no network calls. Write the test first: on the Synthea FHIR
> sample, ~77% of patients with active meds have `med_thermoreg_score > 0` and ~16% have
> `med_combo_raas_diuretic`, each within 10 percentage points.

**`model` — rung 0, then rung 1**
> Read `docs/SPEC.md` §6 and `docs/BUILD_PLAN.md` §3. Implement rung 0 only:
> `leeward/model/priors.py`, `leeward/model/design.py` and `leeward/model/score_prior.py`,
> which draws 400 coefficient vectors from the priors and produces `data/scores.parquet`
> matching the schema, with `p_mean`, `p_lo80`, `p_hi80` and `p_epistemic_share`. No MCMC.
> Test: scores for all five needs exist for every veteran-day, probabilities are in (0,1),
> and `p_lo80 <= p_mean <= p_hi80` everywhere. Stop there; I will ask for rung 1 separately.

**`api` — decision layer**
> Read `docs/SPEC.md` §7. Implement `leeward/decision/allocate.py`: greedy selection
> maximising summed EHA per cost unit under `capacity: dict[str,int]`, at most one action per
> veteran unless `tier == "act_now"` (max 3), with `group_floor` support. Write the test first:
> a five-veteran fixture where the optimum is hand-checkable, plus a property test that
> increasing any capacity never decreases total EHA. Use the fixtures in `data/`, not real scores.

**`ui` — the care-team list**
> Build `ui/src/screens/CareTeam.tsx`. POST `/actions` with `{date, capacity}`, render a ranked
> table (rank, name, tier badge, top driver, EHA), a CapacitySlider (10–100) that refetches on
> release, and a harm-averted counter that animates between values. Read the map base from
> `data/reference/nyc_modzcta.geojson`. Use the stub API and nothing outside `package.json`.

**`demo` — offline safety**
> Write `scripts/clean_clone_test.sh`: clone this repo into a temp dir, create a venv, install,
> run `make demo` with `HTTP_PROXY=http://127.0.0.1:1` set so any network call fails fast, and
> report wall-clock time to first successful `GET /forecast`. It must pass in under 60 seconds
> with no network. Fix whatever it catches.

**`eval` — the Impact bar chart**
> Read `docs/SPEC.md` §11. Implement `leeward/eval/decision_quality.py`: for each day in 91–120
> and K in {20, 40, 80}, select actions with `allocate.py` and with three baselines (rank by age,
> rank by `n_chronic`, random seeded at 0), then compute harm averted as
> `Σ w_k · τ[a,k] · y_true[i,k,t]` over selected veterans. Emit a tidy CSV and a Plotly bar chart.
> Add a test on a tiny fixture where Leeward must beat random.

---

## 10. Checkpoint card

Print this. Check it on the hour.

| T+ | Must be true |
| --- | --- |
| 0:45 | Contracts merged to `main`; `make fixtures` green; six worktrees created |
| 2:30 | `make demo` opens a browser showing a map, a ranked list and a veteran card |
| 6:00 | Real 10,000-veteran cohort; rung-0 scores; real messages with all five mandatory elements |
| 9:30 | Model report screen renders; fairness audit runs; deck v1 exists; **backup video recorded** |
| 11:00 | First timed run-through done; stumbles written down |
| 16:30 | Awake. `make report` finished overnight. |
| 21:00 | Clean-clone test passes offline in under 60 s; rehearsed three times |
| 23:00 | Submitted |

If a checkpoint slips by more than 45 minutes, take the next item off §6 immediately. Do not
negotiate with the cut list.

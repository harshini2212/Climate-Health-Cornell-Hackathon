# Leeward — Care that gets ahead of the weather

**Veteran care continuity under climate events.** Health in Climate AI Hackathon, NYC 2026.

Leeward turns a climate forecast into a ranked action list for VA care teams: **who** to call, **when**, and **why**, given how many calls the team can actually make that day. It predicts daily risk for each veteran, estimates what each outreach action would change, and allocates scarce staff time where it averts the most harm.

Every veteran already has a care team. Leeward tells that team which 40 of its 1,200 patients to reach before Thursday's heat wave, and hands each one a plan that is verifiably from the VA.

> **Synthetic people, real places.** Every neighbourhood-level rate in this repo is real, cited and committed under `data/reference/`; only the people are synthetic. No real patient data is used anywhere, and no code path may ingest PHI.

---

## Contents

- [The problem](#the-problem)
- [What Leeward does](#what-leeward-does)
- [How it works](#how-it-works)
- [Data](#data)
- [Validation](#validation)
- [Demo storyline](#demo-storyline)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Team workflow](#team-workflow)
- [Limits and what we will not claim](#limits-and-what-we-will-not-claim)
- [Documents](#documents)
- [Sources](#sources)

---

## The problem

About 500 New Yorkers die prematurely each summer because of hot weather — roughly 490 heat-exacerbated deaths a year plus about 7 heat-stress deaths, 19 of which came from the June 2025 heat wave alone. Deaths happen overwhelmingly at home, and the rise is driven not by more 95 °F days but by more **"non-extreme hot days," 82 °F up to the extreme-heat threshold** (NYC Health, 2026 Heat Mortality Report). That is why Leeward's heat term hinges at 82 °F: the hyperparameter is a citation.

Veterans carry every heat risk factor the city counts, plus two it does not: deployment exposure (burn pits, PACT Act presumptive conditions) and trauma. And the hazard reaches the care site itself — VA station 630, the Manhattan campus that evacuated ahead of Sandy on 28 October 2012, sits in **hurricane evacuation zone 1**. That is a join in this repo, not a claim.

Heat kills over one to three days, not a week. Smoke, coastal flooding, power outages, and a closed VA facility each hit a different group of veterans on a different timeline. A weekly risk score cannot time a Tuesday call, and a list of 10,000 risks is not a plan.

The hackathon brief names five opportunity areas. Most proposals cover the first three (risk, exposure, prediction). Leeward also covers **04, coordination** (a consented partner export) and **05, verified outreach** (messages a scammer cannot imitate).

## What Leeward does

Leeward answers the three questions a care team asks before a climate event, in order:

| Question | Leeward's answer |
| --- | --- |
| **Who** is at risk? | A daily discrete-time hazard model per veteran per need: breathing flare-up, heat illness, mental-health crisis, treatment or medication gap, and loss of access to care — driven by diagnosis, exposure, housing, social support **and the medication list**. |
| **When** does risk peak? | Distributed lags for heat (0–3 days) and smoke (0–2 days), driven by NWS and AirNow forecasts, so alerts fire two to five days ahead. |
| **What changes if we act?** | A decision layer that computes expected harm averted for every (veteran, action, day) and fills the team's daily capacity greedily, like a knapsack. |

**The prescription says more than the diagnosis.** A diagnosis says a veteran has
hypertension. The medication list says he is on hydrochlorothiazide *and* lisinopril — the
exact pairing CDC names as significantly increasing harm in heat — that he has nine days of
supply left, and that those nine days are arriving by mail through a ZIP that floods. Leeward
resolves every RxNorm code to the VA's own drug classes, scores thermoregulatory risk and
anticholinergic burden, and flags cold-chain and controlled-substance dependence. On the
Synthea sample, **77% of patients on active medications carry at least one drug that impairs
heat response and 16% carry the CDC-named pair.** Leeward flags them for the VA clinical
pharmacist; it never changes a dose.

**Uncertainty is a feature, not decoration.** The posterior splits into risk we are sure about and risk we are unsure about. A veteran with a wide interval (unknown AC status, unknown deployment history) gets a cheap 3-minute check-in call because the value of that information is high. A veteran with a narrow, high interval gets the expensive action: an early refill and a cooling-center ride.

### Tiers

| Tier | Trigger | Action | Who does it |
| --- | --- | --- | --- |
| **Act now** | High hazard, narrow interval, powered equipment or active treatment, or a site-dependent patient whose VA site is down | Care-team call, early refill, backup-power or transport plan, alternate-site booking | VA care team |
| **Find out** | Wide interval, missing fields | 3-minute check-in call to fill the gaps; re-score same day | Care team or volunteer partner |
| **Self-serve** | Medium hazard | Verified text with cooling or clean-air site, refill link, 988 press 1 | Automated via VA channels |
| **Everyday** | Low hazard | Monthly wellness plan: shaded green space, movement, VA Whole Health | Automated |

### Verified outreach and the scam shield (area 05)

Scammers pose as FEMA staff, adjusters, and charities after disasters. Every Leeward message is built so a veteran can tell it apart from a scam:

- Sent only through VA channels the veteran already uses (VEText, My HealtheVet secure message, care-team phone). Leeward never sends from a new number.
- Carries a 4-word verification phrase the veteran can read back to the care team.
- States that the VA will never ask you to pay, wire money, or share bank details.
- Includes the VA fraud line, VSAFE 833-388-7233, and the Veterans Crisis Line: dial 988, press 1.
- High-tier plans ship with a one-page "After the storm" scam card generated from VA and FTC guidance.

Caregiver-addressed messages name the veteran, speak to the caregiver in the second person, and never include diagnoses. Messages to low-asset veterans state what is free and never suggest a paid option.

### Coordination export (area 04)

The care team's action list exports, with consent flags, as a partner sheet: which veterans opted in to a Team Rubicon wellness check, which need a Combined Arms ride, which need a HUD-VASH housing contact. One list, one owner (the VA care team), many hands. Rows without consent are excluded, never redacted.

### Outcome log

Each action records done or not, reached or not, and whether the need occurred. These 20–30 rows a week are exactly what a Bayesian model can absorb without a retrain, which is how the literature-derived priors get tested rather than trusted.

## How it works

```
                 ┌──────────────┐
  NWS / AirNow / │  ingest/     │  snapshot → data/raw/*.parquet (offline-safe)
  FloodNet /     │  hazards.py  │
  emPOWER / HVI  └──────┬───────┘
                        ▼
  Synthea FHIR ─► cohort/  ─► cohort.parquet + truth.json + outcomes.parquet
                        │
                        ▼
                 model/fit.py  (NumPyro NUTS on binomial cells) ─► posterior.nc
                        │
                        ▼
                 model/score.py (posterior × today's hazards) ─► scores.parquet
                        │
                        ▼
                 decision/  (EHA, greedy knapsack, tiers) ─► actions.parquet
                        │
                        ▼
                 api/ (FastAPI) ◄──► ui/ (React + deck.gl)
                        │
                 outreach/ (messages, export, outcome log)
                        │
                 eval/ (recovery, calibration, PPC, SBC, ablations, fairness, decision quality)
```

Forecast in, capacity in, a short verified action list out. Every logged outcome updates the model.

### The Bayesian model

Leeward fits a hierarchical discrete-time hazard model. For each veteran *i*, day *t*, and need *k*:

```
logit h_ikt = α_k + φ_z(i),k + β_kᵀ H_i + γ_k Λ_i + Σ_{l=0..3} δ_kl f(C_z(i),t−l)
            + θ_kᵀ (C_z(i),t ⊗ V_i) + η_k U_i + ξ_i
```

| Term | Meaning | Example |
| --- | --- | --- |
| φ | ZIP effect with an ICAR spatial prior, so neighboring ZIPs share strength | East Harlem borrows from Central Harlem, not from Staten Island |
| H | Health status | Age, COPD, CHF, diabetes, active cancer treatment, heat-sensitive meds |
| Λ | Latent cumulative exposure dose, with measurement error | Burn-pit years × era, observed only through a noisy PACT Act presumptive flag |
| f(C) | Hazard intensity, nonlinear | Piecewise-linear heat index with a hinge at the NYC 82 °F threshold; PM2.5; surge; outage; site down |
| C ⊗ V | Hazard × vulnerability interactions | Smoke × COPD; heat × diuretic; outage × oxygen concentrator; flood × basement; site down × dialysis |
| U | Past care use | ER visits, missed refills, missed appointments |
| ξ | Shared frailty across needs | A veteran prone to one need is prone to others |

Social terms (no caregiver, low assets) enter both as main effects and as interactions with hazards, because a co-resident caregiver can act the same day and a veteran who owns an AC but cannot afford to run it is still at risk.

Five design choices, each tied to something a judge can see:

1. **Daily hazard with distributed lags.** Lag weights follow a smooth random-walk prior, so the model learns the lag shape from a handful of numbers.
2. **Latent exposure with measurement error.** Deployment exposure is never observed exactly. Wide uncertainty in Λ flows into wide uncertainty in risk, which routes that veteran to the Find-out tier.
3. **Weakly informative priors, shown not hidden.** Prior centers come from WTC Registry and PACT Act evidence with wide scales. A slider in the demo halves or doubles the prior scale and re-scores live. Rankings barely move, which is the point.
4. **Fit on sufficient statistics, not rows.** 10,000 veterans × 120 days × 5 needs is 6 million Bernoulli rows. Grouping by ZIP × covariate stratum × day gives binomial cells with an identical likelihood and about 50× fewer rows. NumPyro NUTS on JAX fits in minutes on a laptop. The posterior is cached, so scoring a new forecast is a matrix multiply. **Nothing runs inference inside a request.**
5. **Uncertainty decomposition.** The posterior predictive is split into aleatoric and epistemic parts. Epistemic width is what the decision layer treats as value of information.

**Why not XGBoost.** A tree model would fit the synthetic data as well or better. It cannot carry a latent exposure dose, cannot say how much of a prediction is ignorance, cannot share strength across small ZIPs, and cannot update from 30 logged outcomes without a full retrain.

### The decision layer

For each veteran *i*, action *a*, and day *t*:

```
EHA[i,a,t] = Σ_k  w_k · E_posterior[ h_ikt ] · τ[a,k]
VOI[i,t]   = Σ_k  w_k · sqrt( epistemic_var[i,k,t] )        (for check-in calls)
```

- **w_k** is a clinician-editable severity weight per need (a treatment gap for a chemo patient outweighs a missed wellness text).
- **τ[a,k]** is the fraction of need *k* that action *a* prevents. These are literature priors, stated openly and adjustable in the UI.
- Actions are chosen greedily by EHA per unit cost under the team's capacity (calls, refills, rides, bookings, evacuations). One action per veteran per day, up to three for Act-now veterans. An optional per-borough capacity floor supports the fairness audit.

Drivers shown on a veteran's card come from posterior linear-predictor contributions, rendered as plain phrases such as "dialysis at Manhattan VA while it is closed." No SHAP.

## Data

Real public data describes **places and conditions**. A documented synthetic generator describes **people**. Every borrowed number is cited in `docs/sources.md`, and every simulated column carries a sibling `<col>_synthetic = True` flag in the schema.

### Real layers (all free, ZIP or tract level)

| Layer | What it feeds | Source |
| --- | --- | --- |
| Electricity-dependent Medicare beneficiaries by ZIP | Prior on powered-equipment prevalence; outage scenario | HHS emPOWER public ArcGIS REST service |
| Heat Vulnerability Index (1–5 by neighborhood) | ZIP-level heat prior; home-AC rates | NYC Health HVI |
| Heat mortality facts | Severity weights, pitch numbers | NYC 2026 heat-related mortality report |
| Air quality and smoke | Daily PM2.5 covariate | EPA AirNow API (2023 Canadian-smoke days as replay scenario) |
| Forecasts and alerts | Event trigger, lead time | NWS API (api.weather.gov), no key |
| Street flooding | Flash-flood trips by sensor | FloodNet NYC |
| Stormwater depth and hurricane evacuation zones | Flood × floor interaction; evacuation actions | NYC Open Data |
| Flood and hurricane risk | Long-run hazard prior | FEMA National Risk Index |
| Social vulnerability, chronic disease rates | ZIP random-effect covariates | CDC SVI, CDC PLACES |
| Veteran counts by age, era, disability | Cohort re-homing weights | ACS tables S2101 / S2102 |
| Care and crisis locations | Action plans; SiteDown marker | VA Facilities API, SAMHSA locator, NYC cooling centers |

### Synthetic cohort

Start from the VA's public Synthea release of 10,000 synthetic veteran records (FHIR R4), then:

1. **Re-home** each record to an NYC ZIP, sampling in proportion to ACS veteran counts by age band, and assign the nearest VA facility (dialysis and OTP patients go to Manhattan or Brooklyn, the two sites with those services).
2. **Augment** with fields Synthea lacks: deployment era and burn-pit dose, PTSD severity, home AC (from HVI neighborhood rates), powered equipment (from emPOWER counts), floor of residence, evacuation zone, mobility, caregiver status, income band, and heat-sensitive medications.
3. **Simulate** 120 days of daily outcomes from a known generative model with fixed true coefficients in `data/truth.json`. This is the ground truth the model must recover. The simulator and the model share one design module, so there is no feature-mapping drift.
4. **Hide** 20 percent of the AC and deployment fields at random to create real missingness.

Scenarios are YAML files: `sandy_then_heat` (coastal flood warning, surge, outage, Manhattan VA closed for 45 days, then a three-day heat wave), `ida_flash_flood`, and `smoke_2023`.

## Validation

Because the cohort comes from a generator with known coefficients, Leeward can prove it recovers the truth. All checks render on one "Model report" screen.

| Check | What it shows | Pass bar |
| --- | --- | --- |
| Parameter recovery | True coefficient inside the 90 % posterior interval | ≥ 90 % of parameters |
| Calibration | Predicted 30 % risks happen about 30 % of the time, per need | Expected calibration error < 0.03 |
| Held-out discrimination | Days 91–120 unseen during fit | PR-AUC per need vs. a no-climate baseline; report both |
| Posterior predictive check | Simulated daily counts by ZIP and facility vs. observed | Inside the 90 % band on ≥ 90 % of days |
| Simulation-based calibration | 50 small refits on fresh draws from the prior | Rank histograms roughly flat |
| **Decision quality** | Harm averted at K ∈ {20, 40, 80} calls/day vs. ranking by age, by chronic-condition count, and at random | The Impact slide |
| Ablations | Drop climate terms, lags, interactions, latent dose, spatial prior, SiteDown | Calibration and harm averted per row |
| Fairness audit | Calibration and false-negative rate by borough, HVI band, evacuation zone, income band, caregiver status, race and ethnicity | Flag any group > 20 % relative gap; the gap is shown, never suppressed |

## Demo storyline

Leeward opens with one veteran, not a map.

**"Walter"** (synthetic): 71, Bronx, Gulf War era, COPD on the PACT Act presumptive list, oxygen concentrator at night, takes a diuretic, no AC on record. It is Monday. NWS forecasts a coastal storm Wednesday with outages likely, then a three-day heat wave from Friday. Walter is one of 1,200 patients on his care team's panel.

Screens, in order:

1. **Forecast panel.** NWS alert and AirNow feed for NYC. The model re-scores 10,000 veterans in under a second.
2. **Map.** Expected needs by ZIP for the next 7 days with intervals. The Bronx and East Harlem light up. One click on a ZIP shows why: HVI 5, low AC, high emPOWER count. VA facilities turn red when SiteDown.
3. **Care-team list.** Today's actions, cut at 40 calls. Walter is number 3. His card shows a 62 % (45–78) chance of a treatment gap by Thursday; top drivers: oxygen concentrator × outage, diuretic × heat, no AC. Action: call today, backup-power plan, early refill, cooling-center ride Friday.
4. **Capacity slider.** Drag from 40 to 20 calls and the list re-ranks and the harm-averted counter drops. Drag to 80 and the Find-out tier fills with wide-interval veterans.
5. **The message Walter gets.** VA channel, verification phrase, 988 press 1, VSAFE number, the never-pay line.
6. **Model report.** Recovery plot, calibration curve, harm averted vs. the age-ranked baseline, fairness table by borough.

## Repository layout

```
leeward/
  CLAUDE.md                     # project brief, contracts, rules for every coding session
  Makefile                      # data | cohort | fit | score | demo | report | test
  pyproject.toml
  docs/
    SPEC.md                     # full build spec: contracts, acceptance tests, per-track prompts
    Leeward_Proposal.pdf        # the proposal
    sources.md                  # every cited number with a URL
    slides/
  data/
    raw/                        # gitignored, fetched by `make data`, snapshotted for offline demo
    cohort.parquet  truth.json  outcomes.parquet  posterior.nc
    scores.parquet  actions.parquet  outcome_log.parquet
  scenarios/                    # sandy_then_heat.yaml  ida_flash_flood.yaml  smoke_2023.yaml
  leeward/
    schema.py                   # pydantic + polars schemas: the frozen contracts
    ingest/                     # nws, airnow, floodnet, empower, hvi, stormwater, evac_zones, acs, facilities
    cohort/                     # fhir_reader, rehome, augment, simulate, missingness
    model/                      # priors, design, hazard (NumPyro), fit, score, decompose
    decision/                   # severity, tau, eha, allocate, tiers, value_of_info
    outreach/                   # messages, verify, export, outcome_log
    api/                        # FastAPI: /forecast /scores /veteran /actions /message /log /report /export
    eval/                       # recovery, calibration, ppc, sbc, holdout, ablate, fairness, decision_quality
  ui/                           # React + Vite + deck.gl: Forecast, Map, CareTeam, VeteranCard, Message, Report
  tests/
```

The full contracts (every parquet column, every API route) are in [docs/SPEC.md](docs/SPEC.md) §3 and §9.

## Getting started

**Stack.** Python 3.11 · NumPyro 0.15+ on JAX (CPU) · polars · pandas/pyarrow · ArviZ · FastAPI + pydantic v2 · React 18 + Vite + deck.gl + MapLibre · pytest · Makefile.

```bash
git clone https://github.com/harshini2212/Climate-Health-Cornell-Hackathon.git
cd Climate-Health-Cornell-Hackathon
make setup      # venv + deps
make test
make demo       # must boot offline, with no .env and no API key
```

**The public data is already in the repo.** `data/reference/` holds 21 joined and verified
tables (~4 MB, committed): NYC MODZCTA polygons, the Heat Vulnerability Index, hurricane
evacuation zones, stormwater flood extent, CDC PLACES and SVI, FEMA National Risk Index, HHS
emPOWER, ACS veteran counts, the VHA facility registry, FloodNet events, and real EPA AirNow
PM2.5 for the June 2023 smoke episode. `make sources` re-fetches them; nothing needs an API
key, and `make demo` never touches the network. See [`data/README.md`](data/README.md).

Makefile targets, in pipeline order:

| Target | What it does |
| --- | --- |
| `make sources` | Re-fetch the public data into `data/reference/`. Already done and committed; only needed to refresh. |
| `make fixtures` | Fake-but-correctly-shaped parquets, so every lane can start before the real cohort exists. |
| `make cohort` | Build the synthetic NYC cohort, plant `truth.json`, simulate 120 days of outcomes. |
| `make fit` | Fit the NumPyro model on binomial cells and cache `posterior.nc` (about 5–10 min on 8 CPU cores). Not built yet: the model is at rung 0, so the target refuses with an explanation until `leeward/model/fit.py` lands, and starts fitting the moment it does. |
| `make score` | Score the cohort against the scenario's hazards, then allocate: `scores.parquet` and `actions.parquet`. Runs the rung-0 prior scorer until there is a posterior one. |
| `make demo` | Boot the API and UI. Target: under 60 s from a clean clone, fully offline. |
| `make report` | Run the full evaluation harness, including the fairness audit, into `report/report.json`. |
| `make test` | `pytest -q`. Must pass before any merge. |

> The codebase is being built during the hackathon (19–20 September 2026). Until the Makefile lands, the spec in `docs/SPEC.md` is the source of truth for what each target must do and how it is accepted.

## Team workflow

Two builders, six Claude Code terminals, six git worktrees, one `main`. The full plan —
contract freeze, checkpoints, cut list, per-lane prompts — is in
[`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md).

| Owner | Lanes | Ships |
| --- | --- | --- |
| **Rahul** — runs the live demo | `api`, `ui`, `demo` | A demo that never breaks |
| **Partner** — owns the numbers | `cohort`, `model`, `eval` | Numbers that survive "how do you know?" |

Two rules make two people work like four:

1. **Spine before substance.** By T+2h there is a running end-to-end demo with fake numbers.
   Every hour after that replaces one fake with one real thing. You are never in a state
   where there is nothing to show.
2. **Contracts before parallelism.** The first 45 minutes produce `schema.py`,
   `api/schemas.py` and a fixture generator. After that no lane blocks another, and nobody
   edits a contract file they do not own.

The model is built as a **ladder** — prior-only, then pooled NUTS, then interactions and
SiteDown, then ICAR and the latent dose — each rung writing the same posterior contract. Rung
0 is built first and is the demo's floor, not a fallback.

Rules everyone follows (see [CLAUDE.md](CLAUDE.md)):

- Branch per lane. Merge to `main` through a green `pytest -q`, every 60–90 minutes.
- One task per prompt. Write the acceptance test first. Ask before touching a contract file.
- Never invent a neighbourhood rate that already exists in `data/reference/`.
- Every real number shown anywhere is in `docs/sources.md` with a URL. Synthetic numbers say so.
- A failing fairness audit is displayed, never suppressed.

**Cut in this order:** SBC → ICAR + latent dose → prior slider → partner export →
outcome-log write-back → Ida scenario → deck.gl (fall back to Plotly) → rung-2 interactions.

**Never cut:** Walter's card, the capacity slider, the calibration plot, the SiteDown term,
the verified-message screen.

## Limits and what we will not claim

- Every veteran is synthetic. Every individual-level effect is an assumption we planted and then recovered. Nothing here validates the model on real veterans; it validates that the machinery is correct and the uncertainty is honest, which is the precondition for a VA pilot.
- Eligibility and enrollment are not public. The demo assumes all 10,000 are enrolled with a care team.
- Burn-pit dose is unobserved in reality, so it is a latent variable with era-based priors, not a known number.
- Outage feeds are not archived, so the outage scenario is scripted from a past Con Ed event.
- WTC Registry and PACT Act evidence set prior centers only. The ranking is shown to be robust to halving or doubling them.
- Intervention effects τ are literature priors, adjustable in the UI, not measured here.
- The tool proposes. Clinicians approve every high-tier action. Nothing changes a medication, decides eligibility, or contacts a veteran outside VA-approved channels.

### Next steps after the hackathon

1. Validate priors on de-identified VA data with a VA research partner under IRB, replacing planted truth with real outcomes.
2. Pilot for one summer with one NYC VA medical center care team. Success metric: harm averted at fixed staff time, measured against the prior summer.
3. Connect the Everyday tier to VA Whole Health and NYC Parks shade data so the same model serves veterans on ordinary days.
4. Publish the cohort generator and evaluation harness as open source so other teams can benchmark against a shared synthetic NYC cohort.

## Documents

- [docs/proposal.md](docs/proposal.md) — the full proposal: problem, cohort, model, validation plan, action catalog, demo script, pitch script, risks.
- [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) — the two-person one-day plan: lanes, contract freeze, checkpoints, cut list, model ladder, demo-engineering checklist, per-lane prompts.
- [docs/SPEC.md](docs/SPEC.md) — the build spec: architecture, data contracts, per-module acceptance tests, Claude Code prompts.
- [data/README.md](data/README.md) — the data catalog, and where every cohort rate comes from.
- [docs/sources.md](docs/sources.md) — every cited number with a URL, plus the endpoints that moved.
- [CLAUDE.md](CLAUDE.md) — the project brief every coding session reads first.

## Sources

Full citations, every fetched endpoint, and a record of which upstream APIs moved or started
requiring keys are in [`docs/sources.md`](docs/sources.md). Headline numbers:

| Number | Source |
| --- | --- |
| ~500 premature heat deaths per NYC summer; the 82 °F threshold; 3× heat-stress death rate for Black New Yorkers | NYC Health, 2026 Heat-Related Mortality Report |
| Ida: 3.15 in/hr rain against 1.75 in/hr sewer capacity; 11 basement deaths | NYC Health, flooding and health |
| Manhattan VA evacuated 28 Oct 2012; OTP closed 5 months; ~100 veterans needed guest-dosing | Griffin et al. 2018; Lukowsky et al. 2019 |
| **Station 630 is in evacuation zone 1** | Computed here: VHA facility registry × NYC hurricane evacuation zones |
| **131,195 NYC veterans, 53.5% aged 65+** | Computed here: ACS 2023 5-year B21001 by ZCTA |
| **36,146 electricity-dependent Medicare beneficiaries in NYC** | Computed here: HHS emPOWER |
| **PM2.5 203.5 µg/m³, AQI 254, Queens, 7 June 2023** | Computed here: EPA AirNow daily files |
| Heat-medication mechanisms and the ACE-inhibitor/ARB-plus-diuretic combination | CDC, Heat and Medications — Guidance for Clinicians |
| ~80% of VA outpatient prescriptions delivered by mail; retail emergency refill **excludes** controlled substances | VA CMOP and Pharmacy Disaster Relief Plan |
| VSAFE fraud line 833-388-7233; Veterans Crisis Line 988 press 1 | VA |

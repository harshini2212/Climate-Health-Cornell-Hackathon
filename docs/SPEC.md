# Leeward — Build Spec and Claude Code Plan

*Companion to Leeward_Proposal.md · 19 Sep 2026 · 26 working hours, 4 parallel tracks*

This file is written to be dropped into the repo as `docs/SPEC.md` and pointed to from `CLAUDE.md`. Every section states a contract, an acceptance test, and the Claude Code prompt that builds it.

---

## 0. Ground rules

- **Synthetic only.** No code path may ingest PHI. Every simulated column has a sibling `<col>_synthetic = True`.
- **Contracts are frozen after hour 2.** `cohort.parquet`, `scores.parquet`, `posterior.nc`, and the API schemas. Changing one requires editing this file and pinging all tracks.
- **Nothing runs inference in a request.** `make fit` produces a cached posterior; the API scores from cache.
- **Every real number in the UI or slides is in `docs/sources.md` with a URL.**
- **Cut list is respected in order** (section 12). Walter's card, the capacity slider, the calibration plot, and the SiteDown term are never cut.

---

## 1. Architecture

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

**Stack.** Python 3.11 · NumPyro 0.15+ on JAX (CPU) · polars · pandas/pyarrow · ArviZ · FastAPI + pydantic v2 · React 18 + Vite + deck.gl + MapLibre · pytest · Makefile.

---

## 2. Repo layout

```
leeward/
  CLAUDE.md
  Makefile                      # data | cohort | fit | score | demo | report | test
  pyproject.toml
  docs/
    SPEC.md                     # this file
    sources.md                  # every cited number with URL
    slides/                     # 8 slides
  data/
    raw/                        # gitignored, fetched by `make data`
    cohort.parquet
    truth.json
    outcomes.parquet            # veteran × day × need, simulated
    posterior.nc
    scores.parquet
    actions.parquet
    outcome_log.parquet
  scenarios/
    sandy_then_heat.yaml
    ida_flash_flood.yaml
    smoke_2023.yaml
  leeward/
    schema.py                   # pydantic + polars schemas, the contracts
    ingest/
      nws.py  airnow.py  floodnet.py  empower.py  hvi.py  stormwater.py
      evac_zones.py  acs.py  facilities.py  snapshot.py
    cohort/
      fhir_reader.py            # Synthea FHIR R4 bundles → tidy tables
      rehome.py                 # → NYC ZIP + assigned VA facility
      augment.py                # exposure, PTSD, AC, equipment, floor, evac zone
      simulate.py               # generative model with truth.json
      missingness.py
    model/
      priors.py
      design.py                 # builds X matrices and binomial cells
      hazard.py                 # NumPyro model
      fit.py
      score.py
      decompose.py              # drivers + epistemic share
    decision/
      severity.py               # w_k
      tau.py                    # τ[a,k] priors
      eha.py
      allocate.py               # greedy knapsack under capacity
      tiers.py
      value_of_info.py
    outreach/
      messages.py
      verify.py                 # 4-word phrase
      export.py                 # partner sheet with consent flags
      outcome_log.py
    api/
      main.py  schemas.py  routes_forecast.py  routes_scores.py
      routes_actions.py  routes_report.py  routes_log.py
    eval/
      recovery.py  calibration.py  ppc.py  sbc.py  holdout.py
      ablate.py  fairness.py  decision_quality.py  report.py
  ui/
    src/
      App.tsx
      screens/ Forecast.tsx  Map.tsx  CareTeam.tsx  VeteranCard.tsx
               Message.tsx  Report.tsx
      components/ CapacitySlider.tsx  PriorSlider.tsx  TierBadge.tsx
  tests/
    test_schema.py  test_cohort.py  test_hazard_toy.py
    test_allocate.py  test_messages.py  test_api.py  test_eval.py
```

---

## 3. Data contracts (schema.py)

### 3.1 cohort.parquet — one row per veteran

| column | type | source | notes |
| --- | --- | --- | --- |
| veteran_id | str | synthea | Patient.id |
| age | int | synthea | |
| sex | str | synthea | |
| race, ethnicity | str | synthea | for fairness audit only |
| zip | str | rehome | NYC ZCTA |
| borough | str | rehome | |
| facility_id | str | rehome | nearest of {NY_MANHATTAN, NY_BROOKLYN, NY_BRONX, NY_ST_ALBANS, CBOC_*} |
| lives_alone | bool | synthea SDoH | |
| conditions | list[str] | synthea | SNOMED codes |
| copd, asthma, chf, diabetes, ckd_dialysis, active_cancer_tx, ptsd, depression | bool | derived | |
| pact_presumptive | bool | derived | any PACT respiratory/cancer code |
| n_chronic | int | derived | |
| heat_sensitive_meds | bool | derived | diuretic, anticholinergic, beta-blocker, antipsychotic (RxNorm classes) |
| on_methadone_otp | bool | augment | site-dependent treatment |
| powered_equipment | str | augment | none / oxygen / ventilator / wheelchair / bed / dialysis_home |
| deployment_era | str | augment | vietnam / gulf / post911 / peacetime |
| burn_pit_years | float | augment | latent truth; observed with noise |
| ptsd_severity | int 0–4 | augment | |
| home_ac | bool | augment | from HVI neighborhood AC rate |
| floor | str | augment | basement / ground / upper |
| evac_zone | int 0–6 | augment | 0 = none |
| stormwater_depth_ft | float | augment | moderate-rain scenario, ZIP centroid |
| mobility_impaired | bool | augment | |
| caregiver | str | augment | none / informal_coresident / informal_remote / va_pcafc |
| caregiver_contact_consent | bool | augment | may message caregiver as primary contact |
| income_band | str | augment | low (<150% FPL) / mid / high; from ZIP ACS income and poverty, Synthea SDoH |
| low_assets | bool | augment | cannot self-fund AC use, transport, hotel, med replacement |
| er_visits_12m, missed_refills_12m, missed_appts_12m | int | synthea | |
| *_synthetic | bool | | one per augmented column |
| *_observed | | | nullable copies after missingness |

### 3.2 hazards.parquet — one row per zip × day

`zip, date, heat_index_max_f, hot_day(>=82F), heat_alert, pm25, smoke_alert, flood_watch, flood_warning, flash_flood_emergency, surge_ft, floodnet_trip, outage_frac, site_down: dict[facility_id, bool]`

### 3.3 outcomes.parquet — simulated truth

`veteran_id, date, need ∈ {breathing, heat, mental, treatment_gap, access_loss}, y ∈ {0,1}`

### 3.4 scores.parquet

`veteran_id, date, need, p_mean, p_lo80, p_hi80, p_epistemic_share, driver_1, driver_2, driver_3, driver_1_contrib, driver_2_contrib, driver_3_contrib`

### 3.5 actions.parquet

`date, veteran_id, action, tier, eha, rank, capacity_bucket, rationale, message_id`

### 3.6 outcome_log.parquet

`action_id, veteran_id, date, done, reached, need_occurred, partner_ack, logged_by`

---

## 4. Ingest (Track A, hours 0–2)

Each module: `fetch() -> polars.DataFrame`, `snapshot()` writes `data/raw/<name>.parquet`. `make data` calls all; if the network fails, it loads the snapshot and prints a warning.

| module | endpoint | notes |
| --- | --- | --- |
| nws.py | `https://api.weather.gov/alerts/active?area=NY` and gridpoint forecasts for 5 borough points | no key; parse heat, coastal flood, flash flood, hurricane alerts |
| airnow.py | AirNow API (free key) | daily PM2.5 by ZIP; fallback CSV for June 2023 |
| floodnet.py | floodnet.nyc data API | sensor id, lat/lon, depth series; map sensor → ZIP |
| empower.py | ArcGIS REST FeatureServer (public) | `where=STATE='NY'` , fields `ZIP, POWER_DEPENDENT, OXYGEN, ...`; pages of 2,000 |
| hvi.py | NYC Health portal export | HVI 1–5 by NTA; NTA → ZIP crosswalk |
| stormwater.py | NYC Open Data 9i7c-xyvv (GIS) | rasterize moderate-rain depth to ZIP mean and 90th pct |
| evac_zones.py | NYC Open Data hurricane evacuation zones | zone per ZIP (area-weighted mode) |
| acs.py | Census API S2101 | veteran count by ZCTA × age band |
| facilities.py | VA Facilities API | NYC facilities with lat/lon; add `in_evac_zone` and `flood_depth_ft` by point lookup |

**Acceptance:** `make data` completes offline from snapshots; `tests/test_schema.py` validates every parquet against `schema.py`.

**Claude Code prompt (Track A, first task):**
> Read CLAUDE.md and docs/SPEC.md §3–4. Implement `leeward/ingest/empower.py`: fetch NY ZIP rows from the public HHS emPOWER ArcGIS FeatureServer with paging (2,000/page), keep ZIP, total beneficiaries, and each DME column, write `data/raw/empower.parquet`, and add `snapshot()` fallback. Add a test that loads the snapshot and checks ≥ 150 NYC ZIPs are present and all counts are non-negative. Do not touch other modules.

---

## 5. Cohort generator (Track A, hours 2–10)

### 5.1 fhir_reader.py
Reads Synthea FHIR R4 bundles (the VA release ships CSV and FHIR; use FHIR so the same reader points at a real FHIR base later). Extract Patient, Condition, MedicationRequest, Device, Encounter, Observation (SDoH). Flag `--fhir-base URL --token` for a live server (stretch).

### 5.2 rehome.py
Sample ZIP for each veteran with `P(zip | age_band) ∝ ACS veteran count`. Assign `facility_id` = nearest VA by haversine, except dialysis and OTP patients are assigned to Manhattan or Brooklyn (the two with those services) by proximity.

### 5.3 augment.py — rates to match public figures

| field | how |
| --- | --- |
| deployment_era | by birth year; gulf/post911 share matches ACS era table |
| burn_pit_years (truth) | era = gulf/post911: LogNormal(mean 0.8 yr); else 0. Observed proxy: `pact_presumptive` (true positive 0.6, false positive 0.05) |
| ptsd_severity | ptsd=True → Categorical over 1–4; else 0 |
| home_ac | Bernoulli(HVI-band AC rate: HVI5 0.76 … HVI1 0.95) |
| powered_equipment | Multinomial with ZIP rate = emPOWER count / ACS 65+ population, capped |
| floor | basement 0.06, ground 0.25, upper 0.69 citywide; basement up-weighted ×2 in high stormwater-depth ZIPs |
| evac_zone, stormwater_depth_ft | ZIP lookup |
| mobility_impaired | 0.18 for 65+, 0.08 under 65 |
| on_methadone_otp | 0.01 overall |
| caregiver | 65+: none 0.35, informal_coresident 0.40, informal_remote 0.20, va_pcafc 0.05; under 65: none 0.55, coresident 0.30, remote 0.12, pcafc 0.03; `lives_alone=True` forces none or remote |
| income_band | Bernoulli(low) = ZIP poverty rate × 1.3 (veterans 65+ skew lower), else mid/high 70/30 |
| low_assets | P = 0.7 if income low, 0.2 if mid, 0.05 if high |

### 5.4 simulate.py — the generative truth
Implements exactly the model in §6 with coefficients from `truth.json` (committed). Loops 120 days under a scenario YAML; emits `outcomes.parquet`. **The simulator and the model share `design.py` so there is no feature-mapping drift.**

`truth.json` (excerpt; log-odds):
```json
{
  "alpha": {"breathing": -5.5, "heat": -6.0, "mental": -5.8, "treatment_gap": -5.2, "access_loss": -6.5},
  "beta_age65": {"heat": 0.6, "treatment_gap": 0.3},
  "gamma_lambda": {"breathing": 0.5},
  "delta_heat_lag": {"heat": [0.9, 0.7, 0.4, 0.15]},
  "eps_pm25_lag": {"breathing": [0.8, 0.5, 0.2]},
  "zeta_flood": {"access_loss": 1.2, "treatment_gap": 0.6},
  "kappa_outage": {"treatment_gap": 0.8, "breathing": 0.4},
  "psi_sitedown": {"treatment_gap": 1.5, "access_loss": 0.9},
  "theta": {
    "heat_x_meds": {"heat": 0.7},
    "smoke_x_pact": {"breathing": 0.9},
    "outage_x_equipment": {"treatment_gap": 1.4},
    "flood_x_lowfloor": {"access_loss": 1.3},
    "flood_x_mobility": {"access_loss": 0.8},
    "sitedown_x_sitedependent": {"treatment_gap": 2.0}
  },
  "sigma_social": {
    "no_caregiver": {"treatment_gap": 0.6, "access_loss": 0.7, "heat": 0.5, "mental": 0.4},
    "low_assets": {"heat": 0.5, "access_loss": 0.6, "treatment_gap": 0.3}
  },
  "theta_social_x_hazard": {
    "no_caregiver_x_any_hazard": 0.5,
    "low_assets_x_heat": 0.6,
    "low_assets_x_flood_or_outage": 0.7
  },
  "sigma_zip": 0.4, "sigma_frailty": 0.5
}
```

### 5.5 Scenario YAML

```yaml
# scenarios/sandy_then_heat.yaml
name: sandy_then_heat
days: 120
events:
  - {day: 60, type: coastal_flood_warning, zones: [1,2], lead_days: 3}
  - {day: 63, type: surge, zones: [1,2], surge_ft: 9}
  - {day: 63, type: outage, zips: LOWER_MANHATTAN+ROCKAWAYS, frac: 0.8, duration_days: 4}
  - {day: 63, type: site_down, facility: NY_MANHATTAN, duration_days: 45}
  - {day: 66, type: heat_wave, heat_index_max_f: [96, 99, 97], heat_alert: true}
```
```yaml
# scenarios/ida_flash_flood.yaml
events:
  - {day: 45, type: flash_flood_emergency, lead_days: 0, hour: 20}
  - {day: 45, type: floodnet_trip, zips: QUEENS_FLOODNET_SET, depth_ft: 2.5}
  - {day: 46, type: outage, zips: QUEENS_SUBSET, frac: 0.3, duration_days: 1}
```

**Acceptance:** `make cohort` writes cohort, outcomes and truth; `test_cohort.py` checks group rates (including caregiver and income bands) within ±20 percent of targets and that outcome base rates per need are 0.2–2 percent per day off-event.

**Claude Code prompt (Track A, simulate):**
> Implement `leeward/cohort/simulate.py` using `leeward/model/design.py` to build the linear predictor from `truth.json` and `hazards.parquet`, then draw Bernoulli outcomes for 120 days. Add a test that, with all hazards zeroed, the mean daily rate per need is within 30 percent of `sigmoid(alpha_k)`, and that on `site_down` days the treatment-gap rate for dialysis patients at that facility at least triples.

---

## 6. Model (Track B)

### 6.1 design.py — shared by simulator and model
Builds `X_health (N×p)`, `X_int (N×q per hazard)`, `hazard tensors (Z×T×m)`, `lag stacks`, and **binomial cells**: group by `(zip, stratum, date)` where stratum = the tuple of binary vulnerability flags used in interactions, now including `no_caregiver` and `low_assets` (≤ 256 strata; still ~50× fewer rows than Bernoulli). Output `cells.parquet: zip, stratum_id, date, need, n, y`.

### 6.2 hazard.py — NumPyro

```python
def model(cells, X_strat, H_zip, lags, adj, n_needs=5):
    K = n_needs
    alpha = sample("alpha", Normal(-5.5, 1.0).expand([K]))
    # ICAR ZIP effect per need
    tau_zip = sample("tau_zip", HalfNormal(1.0).expand([K]))
    phi_raw = sample("phi_raw", Normal(0, 1).expand([Z, K]))
    phi = icar_transform(phi_raw, adj) * tau_zip
    # health, latent dose, lags, hazards, interactions (all Normal, weakly informative; see priors.py)
    ...
    # latent burn-pit dose: non-centered
    lam_mu = era_prior_mean[era]; lam_sd = 0.6
    lam_raw = sample("lam_raw", Normal(0,1).expand([S]))
    lam = lam_mu + lam_sd * lam_raw
    sample("pact_obs", Bernoulli(sigmoid(a + b*lam)), obs=pact_flag)   # measurement model
    # distributed lags with RW prior
    delta = sample("delta_heat", Normal(0, 0.5).expand([K,4]))  # + RW smoothness penalty via factor()
    ...
    logit = alpha[need] + phi[zip, need] + X_strat @ beta[:, need] + ... 
    sample("y", Binomial(total_count=n, logits=logit), obs=y)
```

Notes:
- Non-centered everywhere. `init_to_median`, 4 chains × 1,000 warmup × 1,000 samples, `target_accept=0.9`.
- Cells reduce ~6M rows to ~300k; expect 5–10 min on 8 CPU cores. If slower, `num_chains=2` and 500/500.
- Frailty ξ_i is per **stratum**, not per veteran, at cell level (a documented approximation); per-veteran frailty is recovered at score time via the outcome log.

### 6.3 priors.py

| parameter | prior | note |
| --- | --- | --- |
| θ heat×meds | Normal(0.4, 0.5) | wide; WTC/heat-med literature |
| θ smoke×PACT | Normal(0.6, 0.5) | PACT presumptive list |
| θ outage×equipment | Normal(0.8, 0.6) | emPOWER rationale |
| θ flood×lowfloor | Normal(0.8, 0.6) | Ida basement evidence |
| θ sitedown×sitedependent | Normal(1.0, 0.7) | Sandy dialysis/OTP evidence |
| β PTSD | Normal(0.3, 0.3) | WTC mind–body |
| σ no_caregiver (main, per need) | Normal(0.5, 0.4) | clinician input; social-isolation literature |
| σ low_assets (main) | Normal(0.4, 0.4) | NYC heat report: AC owned but not run for cost |
| θ no_caregiver × any hazard | Normal(0.4, 0.4) | shared across hazards, one parameter |
| θ low_assets × heat; × flood/outage | Normal(0.5, 0.4) each | |
| δ, ε lag curves | Normal(0,0.5) + RW(σ=0.2) | smoothness |
| φ | ICAR, τ ~ HalfNormal(1) | |
| `prior_scale_multiplier` | CLI arg 0.5 / 1 / 2 | feeds the demo slider (three cached posteriors) |

### 6.4 score.py
`score(posterior, cohort, hazards_today..+7) -> scores.parquet`. Vectorized over posterior draws (thin to 400). Per veteran per day per need: mean, 10th/90th pct, **epistemic share** = Var over draws of p / (Var over draws + mean p(1−p)).

### 6.5 decompose.py
Driver contributions = each term's posterior-mean contribution to the linear predictor; top-3 by absolute value, rendered as plain phrases (`"dialysis at Manhattan VA while it is closed"`).

**Acceptance:** `test_hazard_toy.py` fits 200 veterans × 30 days in < 60 s with zero divergences after warmup; `make fit` writes `posterior.nc` with r-hat < 1.05 on all parameters.

**Claude Code prompt (Track B, first task):**
> Read docs/SPEC.md §6. Implement `leeward/model/hazard.py` and `fit.py` for the binomial-cell likelihood with alpha, ICAR phi, health betas, one interaction block, and the 4-lag heat curve. Skip the latent dose for now (leave a TODO and a flag). Fit on a 200-veteran toy cohort from `tests/fixtures/`. Report r-hat and divergences. Ask me before changing `design.py`.

---

## 7. Decision layer (Track C)

### 7.1 severity.py — w_k (clinician-editable YAML)
`breathing 3, heat 4, mental 4, treatment_gap 5, access_loss 3`

### 7.2 tau.py — τ[a,k], fraction of need prevented (literature priors; editable)

| action | breathing | heat | mental | treatment_gap | access_loss | cost unit |
| --- | --- | --- | --- | --- | --- | --- |
| care_team_call | 0.25 | 0.30 | 0.35 | 0.40 | 0.20 | call |
| early_refill | 0.15 | 0.10 | 0.05 | 0.60 | 0.00 | refill |
| cooling_center_ride | 0.05 | 0.60 | 0.05 | 0.00 | 0.10 | ride |
| clean_air_room | 0.50 | 0.05 | 0.00 | 0.00 | 0.00 | ride |
| backup_power_plan | 0.20 | 0.00 | 0.05 | 0.50 | 0.10 | call |
| alt_site_booking (dialysis/infusion/OTP) | 0.00 | 0.00 | 0.05 | 0.70 | 0.30 | booking |
| evacuation_assist | 0.05 | 0.10 | 0.10 | 0.30 | 0.75 | evac |
| verified_text | 0.05 | 0.15 | 0.10 | 0.10 | 0.05 | free |
| check_in_call (Find out) | information action; value = expected reduction in epistemic variance × w | call |

### 7.3 eha.py
`EHA[i,a,t] = Σ_k w_k · mean_draws(p[i,k,t,draw]) · τ[a,k]`, plus `VOI[i] = Σ_k w_k · sqrt(epistemic_var[i,k,t])` for check-in calls.

### 7.4 allocate.py
Greedy by EHA per unit cost under `capacity = {call: 40, refill: 200, ride: 15, booking: 20, evac: 8}`; one action per veteran per day unless the veteran is Act-now, then up to 3. Optional `group_floor = {borough: 0.1}` for the fairness floor. Deterministic; returns `actions.parquet` with rank and rationale.

### 7.4b Caregiver routing
If `caregiver != none` and `caregiver_contact_consent`, `care_team_call` and `verified_text` target the caregiver first (τ for those actions +0.10 on treatment_gap and access_loss, because a co-resident can act same-day). If `caregiver == none`, `verified_text` τ is halved and `care_team_call` / `evacuation_assist` are preferred; `assign_buddy` becomes available (τ access_loss 0.35, cost unit `partner_slot`). If `low_assets`, `cooling_center_ride` and `evacuation_assist` are booked, not suggested, and `heap_application` is added as a 5-day-out action (τ heat 0.30 over the season).

### 7.5 tiers.py
Act-now: p_mean ≥ 0.25 on any need with w ≥ 4 and epistemic share < 0.4, or any site-dependent × SiteDown, or (`no_caregiver` and `powered_equipment != none` and outage forecast). Find-out: epistemic share ≥ 0.4 and p_mean ≥ 0.10. Self-serve: p_mean 0.05–0.25. Everyday: rest.

**Acceptance:** `test_allocate.py` hand-checkable 5-veteran case; capacity never exceeded; raising capacity never lowers total EHA.

**Claude Code prompt (Track C):**
> Implement `leeward/decision/allocate.py`: greedy selection maximizing summed EHA per cost unit under `capacity: dict[str,int]`, at most one action per veteran unless tier == "act_now" (max 3). Add `group_floor` support. Write a 5-veteran test where the optimum is obvious, and a property test that increasing any capacity never decreases total EHA.

---

## 8. Outreach (Track C)

- `messages.py`: templates per tier and hazard. Every message contains: channel tag (`VEText` / `MHV` / `care_team_phone`), a 4-word verification phrase from `verify.py` (deterministic per veteran-day from a seed; word list of 512 common words), the line *"The VA will never ask you to pay, wire money, or share bank details,"* `VSAFE 833-388-7233`, and `Veterans Crisis Line: dial 988, press 1`. Flood messages add the evacuation center with step-free access and a "pack list" (meds, equipment, chargers, IDs). Caregiver-addressed messages name the veteran, state the plan in second person to the caregiver, and never include diagnoses. Low-assets messages state what is free (HEAP, cooling centers, emergency refill voucher, VA transport) and never suggest a paid option.
- `export.py`: partner sheet CSV with `consent_partner_check, consent_ride, consent_housing` flags; rows without consent are excluded, never redacted.
- `outcome_log.py`: append-only parquet; `POST /log` writes it; `score.py` reads it to update per-veteran frailty (simple Beta-Binomial update on top of the cached posterior).

**Acceptance:** `test_messages.py` asserts every rendered message contains all five mandatory elements; no message contains a URL shortener or a phone number not in the allow-list.

---

## 9. API (Track C)

| route | returns |
| --- | --- |
| `GET /forecast?scenario=&day=` | hazards for the next 7 days by ZIP and facility |
| `GET /scores?date=&need=` | ZIP and facility aggregates with intervals |
| `GET /veteran/{id}?date=` | card: p per need, interval, epistemic share, drivers, tier |
| `POST /actions` body `{date, capacity, group_floor?, prior_scale?}` | ranked action list + total EHA + baselines' EHA |
| `GET /message/{action_id}` | rendered message |
| `POST /log` | outcome log row |
| `GET /report` | eval JSON for the Model report screen |
| `GET /export?date=` | partner sheet CSV |

Stub with fake data by hour 2 so Track D can build against it.

---

## 10. UI (Track D)

Screens, in demo order: Forecast → Map → Care team list → Veteran card → Message → Model report.

- **Map:** deck.gl `GeoJsonLayer` of NYC ZIPs, fill = expected need count for selected need and day; `IconLayer` for VA facilities, red when SiteDown; toggle for evacuation zones and FloodNet sensor trips. Fallback: static Plotly choropleth.
- **Care team list:** table cut at capacity; tier badge; EHA; drivers as chips. **CapacitySlider** re-posts `/actions` on change, animates the harm-averted counter and the baseline bars.
- **Veteran card:** 5 needs as interval bars (10–90 pct) with epistemic share shaded; drivers in plain language; action plan; "Why this tier" line.
- **PriorSlider:** 0.5× / 1× / 2× prior scale; swaps between three cached posteriors.
- **Model report:** recovery dot-whisker, reliability curves, harm-averted bar chart vs baselines, ablation table, fairness table.

**Acceptance:** `make demo` boots API + UI in < 60 s from a clean clone; every screen renders with the stub API; slider round-trip < 300 ms.

**Claude Code prompt (Track D):**
> Build `ui/src/screens/CareTeam.tsx`: fetch `POST /actions` with `{date, capacity}`, render a ranked table (rank, name, tier badge, top driver, EHA), a `CapacitySlider` (10–100) that re-fetches on release, and a harm-averted counter that animates between values. Use the stub API. No external UI kit beyond what is in package.json.

---

## 11. Evaluation harness (Track B, overnight)

| script | output | pass bar |
| --- | --- | --- |
| recovery.py | `report/recovery.json` (truth vs 90 pct interval per parameter) | ≥ 90 pct covered |
| calibration.py | reliability per need on days 91–120 | ECE < 0.03 |
| holdout.py | PR-AUC per need vs no-climate baseline | report both |
| ppc.py | daily counts by ZIP and by facility vs 90 pct band | ≥ 90 pct of days inside |
| sbc.py | 50 refits on 500-veteran draws from prior | rank histograms; runs overnight |
| ablate.py | drops climate / lags / interactions / latent dose / ICAR / SiteDown; calibration + harm averted per row | table |
| decision_quality.py | harm averted at K ∈ {20,40,80} for Leeward vs age / chronic-count / random | the Impact bar chart |
| fairness.py | ECE and FNR by borough, HVI band, evac zone, income band, caregiver status, race/ethnicity | show gaps; flag > 20 pct relative |
| report.py | assembles `report/report.json` for `GET /report` | |

**Claude Code prompt (Track B, eval):**
> Implement `leeward/eval/decision_quality.py`: for each day in 91–120 and K in {20,40,80}, select actions with `allocate.py` and with three baselines (rank by age, rank by n_chronic, random with seed), then compute harm averted = Σ w_k · τ[a,k] · y_true[i,k,t] over selected veterans. Output a tidy CSV and a Plotly bar chart. Add a test on a tiny fixture where Leeward must beat random.

---

## 12. Schedule, tracks and cut list

**Hours (Sat 13:00 → Sun 15:00, sleep 01:00–07:00).**

| Hour | Track A: data + cohort | Track B: model + eval | Track C: decision + API + outreach | Track D: UI + pitch |
| --- | --- | --- | --- | --- |
| 0–2 | `schema.py`; ingest empower, hvi, nws, stormwater, evac zones; snapshots | `hazard.py` on toy cohort, end-to-end NUTS | Stub API matching contracts | Skeleton app on stub; ZIP map renders |
| 2–6 | `fhir_reader`, `rehome`, `augment`; cohort of 10k | `design.py` cells; full fit; `posterior.nc` | `eha`, `allocate`, `tiers` on stub scores | Care-team list, veteran card, capacity slider |
| 6–10 | `simulate.py`; three scenario YAMLs; missingness | `score.py`, `decompose.py`; latent dose block | Real scores wired; `/actions`; messages; verify | Message screen; scam card; forecast panel; SiteDown marker |
| 10–16 (overnight jobs) | `sources.md`; data README | recovery, calibration, PPC, holdout, ablations; **SBC job**; 0.5×/2× prior fits | Baselines; decision-quality inputs; export; outcome log | Model report page; slides v1 |
| 16–22 | Freeze cohort; tag `v1` | Fairness audit; prior-slider posteriors | Bug bash; `make demo` clean-clone test | Rehearse ×3; record 90-s video; two-pager |
| 22–26 | Buffer | Final numbers into slides | Buffer | Final rehearsal; submission |

**Stretch (only if Track A is done by hour 10):** `--fhir-base` pointed at VA Lighthouse or Oracle Health sandbox; 200 synthetic patients pulled live to prove the path.

**Cut list, in order:** SBC → ICAR (fall back to independent ZIP effects) → prior slider → partner export → deck.gl (fall back to Plotly) → Ida scenario (keep Sandy-then-heat). **Never cut:** Walter's card, the capacity slider, the calibration plot, the SiteDown term, the verified-message screen.

---

## 13. CLAUDE.md (drop-in)

```markdown
# Leeward — veteran care continuity under climate events

## What this is
Bayesian daily-hazard model + capacity-aware decision layer for VA care teams, for the
Health in Climate AI Hackathon NYC 2026. Synthetic data only. Never write code that could
ingest real PHI. Full spec: docs/SPEC.md — read the section for your track before coding.

## Contracts (frozen; change only by editing docs/SPEC.md §3 and telling all tracks)
- data/cohort.parquet, hazards.parquet, outcomes.parquet, scores.parquet, actions.parquet,
  outcome_log.parquet: columns in leeward/schema.py.
- data/posterior.nc: ArviZ InferenceData; var names match leeward/model/priors.py.
- API bodies/responses: pydantic models in leeward/api/schemas.py.
- leeward/model/design.py is shared by the simulator and the model. Do not fork it.

## Stack
Python 3.11, NumPyro + JAX (CPU), polars, FastAPI, React + Vite + deck.gl. Makefile targets:
data | cohort | fit | score | demo | report | test. Never run inference inside a request.

## Rules
- Every real number shown anywhere is in docs/sources.md with a URL. Synthetic numbers say so.
- Drivers come from posterior contributions; no SHAP.
- Fairness audit runs in `make report`; a failing audit is displayed, never suppressed.
- Every outreach message includes: VA channel tag, 4-word verification phrase,
  "The VA will never ask you to pay, wire money, or share bank details",
  VSAFE 833-388-7233, and "Veterans Crisis Line: dial 988, press 1".
- `pytest -q` passes before any merge; `make demo` boots in < 60 s from a clean clone.
- One task per prompt; write the acceptance test first; ask before touching a contract file.

## Tracks
A data/cohort · B model/eval · C decision/api/outreach · D ui/docs. Branch per track,
merge through Makefile targets, never by hand-copying files.
```

---

## 14. Working with Claude Code, per session

1. Open with: `Read CLAUDE.md, then docs/SPEC.md §<track section>. Run make test. Summarize what exists and what is missing for my next task.`
2. One task per prompt. State the acceptance test in the prompt. Prefer "add a test, then make it pass."
3. For anything touching `schema.py`, `design.py`, or `api/schemas.py`: `Propose a plan and the diff to docs/SPEC.md first; do not edit until I say go.`
4. Every 2 hours: `Run make test and make demo; report anything red; do not fix unrelated failures.`
5. Sunday 09:00: `Do a clean-clone test in /tmp: git clone, make data (offline), make demo. Report time to boot.`
6. Last hour: `Freeze. Only fix crashes. Produce docs/sources.md from every URL in the repo and confirm each number in ui/ appears there.`

---

## 15. Definition of done

- `make demo` from a clean clone boots in < 60 s and runs offline.
- Both scenarios play end to end; Walter's card, the capacity slider and the SiteDown marker work.
- `report/report.json` shows recovery ≥ 90 pct, ECE < 0.03, harm averted vs baselines, ablation table, fairness table.
- Every message passes `test_messages.py`.
- `docs/sources.md` complete; two-pager, 90-s video and 8 slides in `docs/`.

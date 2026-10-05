# Leeward — Technical Specification

*Companion to [proposal.md](proposal.md) · 19 Sep 2026 · v3*


> **Ingest is done.** §4 was four hours of work; it is now a `data/reference/` directory committed to the repo. See [`data/README.md`](../data/README.md).

Every section states a contract and the acceptance test that holds it.

---

## 0. Ground rules

- **Synthetic only.** No code path may ingest PHI. Every simulated column has a sibling `<col>_synthetic = True`.
- **Contracts change only through this file.** `cohort.parquet`, `scores.parquet`, `posterior.nc`, and the API schemas. A PR that changes one edits §3 in the same change.
- **Nothing runs inference in a request.** `make fit` produces a cached posterior; the API scores from cache.
- **Every real number in the UI or slides is in `docs/sources.md` with a URL.**
- **Geography key is `(region_id, geo_id)`.** A region is a `regions/<id>.yaml`
  (`leeward/geo/region.py`): its geography unit, the subregions fairness floors are drawn
  over, its reference tables and its map. `geo_id` is that region's unit — in `nyc`, one of
  NYC's 178 Modified ZCTAs, so NYC's `geo_id` is what this file used to call `modzcta` —
  and it means nothing without the `region_id` beside it. Never `zip`, `zcta` or NTA.
- **No API keys.** A clean clone with no `.env` must produce a working demo. Three upstream APIs now want keys; `docs/sources.md` §3 lists the keyless replacements the build uses.

---

## 1. Architecture

```
                 ┌──────────────┐
  NWS / AirNow /  │ scripts/     │  DONE: 17 keyless fetchers →
  FloodNet / NRI  │ fetch_       │  data/reference/*.parquet (committed, ~4 MB)
  emPOWER / HVI / │ sources.py   │  + manifest.json with url, rows, sha256
  PLACES / SVI /  └──────┬───────┘
  ACS / VHA              │
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
  scripts/
    fetch_sources.py            # DONE: vendors every public source into data/reference/
    make_fixtures.py            # fake-but-correctly-shaped parquets for every contract table
    clean_clone_test.sh
  data/
    README.md                   # the data catalog: what each file is, and the augment priors
    reference/                  # COMMITTED. 21 tables + manifest.json + nyc_modzcta.geojson
    raw/                        # gitignored: synthea, stormwater GIS, ACS summary file
    cohort.parquet
    truth.json
    outcomes.parquet            # veteran × day × need, simulated
    posterior.nc
    scores.parquet
    actions.parquet
    outcome_log.parquet
  regions/
    nyc.yaml                    # NYC as a region: unit, boroughs, reference tables, map
  scenarios/
    sandy_then_heat.yaml
    ida_flash_flood.yaml
    smoke_2023.yaml
  leeward/
    schema.py                   # pydantic + polars schemas, the contracts
    geo/
      region.py                 # a region yaml -> Region; ref() reads its tables as geo_id
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

**Regions (changed in the region-abstraction PR).** `modzcta` was renamed `geo_id`
everywhere, and `region_id` (a `regions/<id>.yaml` id, e.g. `nyc`) leads the key of
cohort, hazards, scores and actions:

| table | key |
| --- | --- |
| cohort | `region_id, veteran_id` |
| hazards | `region_id, geo_id, date` |
| scores | `region_id, veteran_id, date, need` |
| actions | `region_id, date, veteran_id, action` |

`schema.read()` still loads a file written before regions: it renames `modzcta` to `geo_id`
and reads a missing `region_id` as `nyc`, logging one INFO line per table it aliases.
`schema.write()` refuses a frame that still says `modzcta`. `borough` keeps its name; its
values are the row's region's `subregions` (NYC: the five boroughs), checked per row.
Committed `data/reference/` files keep their sources' column names; the region yaml names
each table's unit column, and `region.ref()` renames it to `geo_id` on read.

`allocate()` serves one region per call, since capacity is one care team's day; a cohort
that spans two regions is refused. NYC action ids are unchanged; another region's are
salted with its id.

### 3.1 cohort.parquet — one row per veteran

| column | type | source | notes |
| --- | --- | --- | --- |
| veteran_id | str | synthea | Patient.id |
| age | int | synthea | |
| sex | str | synthea | |
| race, ethnicity | str | augment | **from ACS B03002** (`acs_race_by_zcta.parquet`): one joint draw from the veteran's own ZIP composition, so P(race, ethnicity \| ZIP) is measured and the individual assignment is synthetic. race ∈ White / Black / Asian / Other (AIAN, NHPI, some other race and multiracial folded in); ethnicity ∈ Hispanic / Non-Hispanic (Hispanic is an origin of any race). For fairness audit only |
| region_id | str | build | `regions/<id>.yaml`, e.g. `nyc`. Leads the key. |
| geo_id | str | rehome | The region's unit; in `nyc` a Modified ZCTA, one of 178. **The join key everywhere, with `region_id`.** |
| borough | str | rehome | The region's subregion (NYC: borough); values from the region yaml |
| facility_id | str | rehome | nearest of {NY_MANHATTAN, NY_BROOKLYN, NY_BRONX, NY_ST_ALBANS, CBOC_*} |
| lives_alone | bool | synthea SDoH | |
| conditions | list[str] | synthea | SNOMED codes |
| copd, asthma, chf, ckd_dialysis | bool | derived | PLACES per ZIP (dialysis: VA ESRD × emPOWER shape) |
| diabetes | bool | synthea | **Track A:** type 2 diabetes from the same Synthea veteran, tilted to PLACES `diabetes_crudeprev` per ZIP, so insulin always arrives with its diagnosis |
| ptsd, depression, active_cancer_tx | bool | synthea | **Track A:** from one whole Synthea v4.0.0 veteran (`synthea_veteran_profiles.parquet`), drawn within sex × ACS age band and tilted to cited rates (§5.3): ptsd to VA NCPTSD by era, depression to PLACES per ZIP. active_cancer_tx is untilted: open malignancy + chemo/radiation in the last 365 days |
| suicide_risk | bool | synthea | **New (Track A).** "At increased risk for suicide" or "suicidal thoughts" finding in the record's 10-year history (`veteran_ptsd`, `veteran_self_harm`). The run has no suicide attempt by a living patient, so attempts are not a flag |
| substance_use_disorder | bool | synthea | **New (Track A).** Open alcohol, opioid or drug-use disorder (`veteran_substance_abuse_*`) |
| homeless | bool | synthea | **New (Track A).** Open "Homeless (finding)" (`homelessness`). Forces `home_ac=False`, `floor=ground`, `lives_alone=True` |
| pact_presumptive | bool | derived | any PACT respiratory/cancer code |
| n_chronic | int | derived | |
| med_rxcuis | list[str] | synthea | RxNorm codes active at the reference date, from the **same** Synthea veteran as the diagnoses |
| va_drug_classes | list[str] | derived | via `va_drug_class_members.parquet`; the VA's own 576-class taxonomy |
| n_active_meds | int | derived | polypharmacy count |
| med_thermoreg_score | float | derived | Σ `weight` over heat-mechanism classes in `med_climate_risk.csv` |
| acb_score | int | derived | anticholinergic burden, 0–3 per drug summed; ≥ 3 is the clinical threshold |
| med_combo_raas_diuretic | bool | derived | **the combination CDC names by name**; ACE-i or ARB plus a diuretic |
| med_renal_triple | bool | derived | NSAID on top of a diuretic and a RAAS agent |
| med_cold_chain | bool | derived | insulin or other refrigerated product; drives the outage interaction |
| med_controlled | bool | derived | opioid, benzodiazepine, barbiturate or stimulant. **Excluded from the VA retail emergency refill benefit.** |
| med_narrow_ti | bool | derived | lithium, warfarin, levothyroxine, antiarrhythmics, anticonvulsants |
| mail_order_pharmacy | bool | augment | **synthetic**, base rate 0.80 from VA's published CMOP share |
| days_supply_remaining | int | augment | **synthetic**; 30-day window fills, 90-day mail fills, uniform phase |
| on_methadone_otp | bool | augment | site-dependent treatment |
| powered_equipment | str | augment | none / oxygen / ventilator / wheelchair / bed / dialysis_home |
| deployment_era | str | augment | vietnam / gulf / post911 / peacetime |
| burn_pit_years | float | augment | latent truth; observed with noise |
| ptsd_severity | int 0–4 | augment | |
| home_ac | bool | augment | synthetic; no public per-person source |
| floor | str | augment | basement / ground / upper |
| evac_zone | int 0–6 | augment | 0 = none |
| stormwater_depth_ft | float | augment | moderate-rain scenario, ZIP centroid |
| mobility_impaired | bool | augment | |
| caregiver | str | augment | none / informal_coresident / informal_remote / va_pcafc |
| caregiver_contact_consent | bool | augment | may message caregiver as primary contact |
| income_band | str | augment | low / mid / high; from CDC SVI `EP_POV150` per tract → MODZCTA |
| low_assets | bool | augment | **from PLACES `shututility_crudeprev`** — cannot self-fund AC use, transport, hotel, med replacement |
| transport_barrier | bool | augment | **from PLACES `lacktrpt_crudeprev`**; gates suggest-vs-book for rides |
| er_visits_12m, missed_refills_12m, missed_appts_12m | int | synthea | |
| *_synthetic | bool | | one per augmented column |
| *_observed | | | nullable copies after missingness |

### 3.2 hazards.parquet — one row per region × geo_id × day

`region_id, geo_id, date, heat_index_max_f, hot_day(>=82F), heat_alert, pm25, smoke_alert, flood_watch, flood_warning, flash_flood_emergency, surge_ft, evac_zone_ordered, floodnet_trip, stormwater_flooded_frac, outage_frac, mail_delivery_disrupted`

`mail_delivery_disrupted` is set by the scenario when a ZIP is flooded, evacuated or in a
sustained outage. Four in five VA prescriptions arrive by mail, so this is not a minor term.

Plus `site_status.parquet`, one row per `facility_id × date` with `site_down: bool` — a
separate table rather than a dict column, because 14 facilities × 120 days is small and a
dict column does not survive a parquet round-trip cleanly.

The 82 °F hot-day threshold is from NYC Health's 2026 mortality report, not a tuning choice.

### 3.3 outcomes.parquet — simulated truth

`veteran_id, date, need ∈ {breathing, heat, mental, treatment_gap, access_loss}, y ∈ {0,1}`

### 3.4 scores.parquet

`region_id, veteran_id, date, need, p_mean, p_lo80, p_hi80, p_epistemic_share, p_gap_lo, p_gap_hi, driver_1, driver_2, driver_3, driver_1_contrib, driver_2_contrib, driver_3_contrib`

`p_gap_lo` and `p_gap_hi` are nullable, and null means the record has no gap — which is not
the same as a gap that would not move the number, and §7.5 needs to tell those apart. Where
they are set, `p_gap_lo ≤ p_mean ≤ p_gap_hi` always: all three are averaged over the same
posterior draws, because taking the ends at the mean coefficients instead puts `p_mean`
outside its own range at these probabilities (Jensen) and breaks every threshold the tier
compares them against.

### 3.5 actions.parquet

`action_id, region_id, date, veteran_id, action, tier, eha, rank, lead_days, capacity_bucket, rationale, owner, message_id`

**`date` is the do-by day, not the day the risk lands.** An action has a day it must be done
by — the alternate dialysis site for Wednesday's surge has to be booked Monday — so
`lead_days` (from `lead_days` in `leeward/decision/tau.yaml`, per docs/proposal.md §6) gives
the gap and the risk day is `date + lead_days`. Capacity is consumed on `date`, so one work
day's forty calls cover its own risk, the bookings two days out and the refills five days out
together. An action whose do-by day has already passed is not offered; `compare()` returns
how many, and `POST /actions` reports it as `n_too_late`.

### 3.6 outcome_log.parquet

`action_id, veteran_id, date, done, reached, need_occurred, partner_ack, logged_by`

---

## 4. Ingest — **already done**

This section used to describe nine modules to be written in the first two hours. They are
written, they have been run, and their output is committed. `leeward/ingest/` is no longer
on the critical path; if you build it at all, build it as a thin re-fetch wrapper.

`scripts/fetch_sources.py` holds 18 fetchers, every one keyless, each writing one table into
`data/reference/` plus a row in `manifest.json` recording url, rows, bytes and a sha256
prefix. The catalog and the column meanings are in [`data/README.md`](../data/README.md).

```bash
python scripts/fetch_sources.py --list      # what exists
python scripts/fetch_sources.py             # refresh the small keyless sources
python scripts/fetch_sources.py --heavy     # + stormwater GIS, ACS summary file, Synthea
```

**What you get, joined and checked:**

| File | Rows | Use |
| --- | --- | --- |
| `nyc_modzcta.parquet` / `.geojson` | 178 | Join key and the deck.gl map base |
| `hvi_by_zcta.parquet` | 184 | Heat prior, 1–5 |
| `evac_zone_by_modzcta.parquet` | 178 | Surge exposure, area-weighted |
| `stormwater_by_modzcta.parquet` | 178 | Pluvial flood exposure |
| `places_zcta_nyc.parquet` | 186 | **Every augment prior** — see §5.3 |
| `empower_ny_zip.parquet` | 1,702 | Powered-equipment prior |
| `acs_veterans_by_zcta.parquet` | 212 | Re-homing weights by age band |
| `va_facilities_nyc_hazard.parquet` | 14 | **The SiteDown input** |
| `airnow_pm25_nyc_smoke2023.parquet` | 124 | The smoke replay scenario |
| `floodnet_events.parquet` | 3,269 | Observed flood events, real depths |
| `svi_nyc_tract.parquet`, `fema_nri_nyc_tract.parquet` | 2,324 each | Context and fairness strata |
| `nws_forecast_nyc.parquet` | 70 | Forecast panel; re-run on demo morning |

**Acceptance:** `pytest -q tests/test_reference.py` loads every file in
`data/reference/manifest.json`, asserts the row count matches, and asserts `modzcta` joins
cleanly across the five ZIP-level tables. `make demo` must pass with the network off.

**Still to write, and it is small:** `leeward/ingest/hazards.py`, which assembles
`hazards.parquet` (region × geo_id × day) and `site_status.parquet` (facility × day) by combining the
reference tables with a scenario YAML. That is one module, not nine.


---

## 5. Cohort generator

### 5.1 fhir_reader.py
Reads Synthea FHIR R4 bundles (the VA release ships CSV and FHIR; use FHIR so the same reader points at a real FHIR base later). Extract Patient, Condition, MedicationRequest, Device, Encounter, Observation (SDoH). Flag `--fhir-base URL --token` for a live server (stretch).

### 5.2 rehome.py
Sample ZIP for each veteran with `P(zip | age_band) ∝ ACS veteran count`. Assign `facility_id` = nearest VA by haversine, except dialysis and OTP patients are assigned to Manhattan or Brooklyn (the two with those services) by proximity.

### 5.3 augment.py — draw every rate from a real per-ZIP source

The previous version of this table invented most of these numbers. CDC PLACES publishes them
per ZCTA, so do not invent them. `data/reference/places_zcta_nyc.parquet` is already joined to
MODZCTA. **A rate that exists in `data/reference/` must be read, not assumed.**

| field | draw from | column |
| --- | --- | --- |
| ZIP sampling weight | `acs_veterans_by_zcta.parquet` | `P(geo_id \| age_band) ∝ vet_<band>` |
| `mobility_impaired` | PLACES | `mobility_crudeprev` |
| `caregiver == none` | PLACES | `emotionspt_crudeprev` (lacks social/emotional support), tempered by `loneliness_crudeprev` |
| `low_assets` | PLACES | `shututility_crudeprev` — the measured "owns an AC, cannot run it" |
| `transport_barrier` | PLACES | `lacktrpt_crudeprev` |
| `copd`, `asthma` base rates | PLACES | `copd_`, `casthma_crudeprev` (`depression` and `active_cancer_tx` moved to Synthea in Track A; see below) |
| `powered_equipment` | emPOWER ÷ ACS 65+ | `dme_power_dependent`, `dme_oxygen`, `dme_esrd_dialysis`, capped |
| `income_band` | SVI tract → MODZCTA | `EP_POV150` |
| `evac_zone` | `evac_zone_by_modzcta.parquet` | `evac_zone_min`, `evac_frac_z1..z7` |
| `stormwater_flooded_frac` | `stormwater_by_modzcta.parquet` | |
| HVI band | `hvi_by_zcta.parquet` | `hvi` |

The gradient these produce, across NYC's 178 MODZCTAs grouped by HVI band, is monotone in
every column — which is a free sanity check that the joins are right:

| HVI | Utility shutoff | Lacks emot. support | Mobility diff. | Lacks transport | COPD |
| --- | --- | --- | --- | --- | --- |
| 1 | 5.3% | 24.8% | 9.3% | 6.0% | 3.9% |
| 3 | 8.9% | 30.8% | 13.2% | 10.1% | 5.3% |
| 5 | 17.2% | 35.0% | 18.8% | 16.6% | 6.7% |

#### Medication, derived not invented

Synthea already puts an RxNorm code on every `MedicationRequest`. `cohort/medications.py`
resolves each to a VA drug class with `va_drug_class_members.parquet` (committed, so no RxNav
call at demo time) and then joins `med_climate_risk.csv` for the mechanism and weight:

```
classes(drug)            = the drug's MOST SPECIFIC VA class only     # see the note below
med_thermoreg_score      = Σ weight    over classes where hazard == 'heat'
acb_score                = Σ acb       over all classes          # ACB scale, >=3 is meaningful
med_combo_raas_diuretic  = any(CV800, CV805) and any(CV70*)      # the CDC-named combination
med_renal_triple         = med_combo_raas_diuretic and any(MS101, MS102)
med_cold_chain           = any(cold_chain == 1)                  # insulin -> outage term
med_controlled           = any(controlled == 1)                  # excluded from retail refill
med_narrow_ti            = any(narrow_ti == 1)                   # lithium, warfarin, insulin
```

**Most specific class wins.** RxNav puts a drug in its VA class *and* that class's parent:
hydrochlorothiazide is CV701 THIAZIDES and CV700 DIURETICS, insulin is HS501 and HS500,
alprazolam is CN302 and CN300. Summing over both scores one mechanism twice — doubling the
weight and the ACB of every parented drug. So each drug contributes its most specific class
and nothing else, which is also the class a pharmacist would say out loud. The hierarchy is
**not hand-written**: it is read off the crosswalk, where a child class's members are a
strict subset of its parent's (27 such pairs, all clean).

Expected prevalence among the **77** sample patients who have an active medication,
measured and reproduced in `tests/test_medications.py`: **77 percent** on ≥1 crosswalk
medication of any hazard, **65 percent** on a heat-mechanism one, **16 percent** on the
CDC-named pair, 10 percent controlled, 9 percent cold-chain, 8 percent narrow-TI, 12 percent
at ACB ≥ 3. (An earlier draft of this paragraph put 77 percent on the heat row and 5 percent
on ACB; both are corrected here and pinned by the test.) The veteran 65+ cohort comes out
higher on all of them — 72 percent heat-impairing, 18 percent on the CDC pair, 11 percent
cold-chain — and a test asserts the direction. If it ever comes out lower, the class mapping
is broken.

Prescriptions reach a synthetic veteran by bootstrap: `cohort/medications.py` draws a whole
active list from `synthea_med_profiles.parquet` (109 rows, one per bundle, committed)
stratified by age band, 18–54 against 55+, because the sample's burden triples at 55+. Whole
lists, so real co-prescribing survives; never assembled drug by drug.

#### Track A: Synthea's veteran modules (changed in the Track A cohort PR)

`ptsd`, `depression`, `active_cancer_tx`, `suicide_risk`, `substance_use_disorder`,
`homeless` and `med_rxcuis` come from **one whole Synthea patient** per veteran, drawn from
`synthea_veteran_profiles.parquet` within the veteran's own sex × ACS age band
(`cohort/synthea.py`). The profiles are a pinned Synthea v4.0.0 run (seed 0, reference and
end date 2026-01-01, 8,000 living New York adults, `veteran_population_override=true`;
exact arguments in `synthea.RUN_ARGS` and `docs/sources.md`). Where a veteran lives, their age
and sex are still ACS (§5.2); Synthea supplies only the clinical record. Because diagnoses
and prescriptions now belong to the same patient, the known gap below (a cold-chain or
psychiatric medication independent of the diagnosis list) closes for these conditions.

**Weighting.** Synthea's own rates are off (PTSD in 1.6% of records, substance use disorder
in 26.8%) and carry no ZIP gradient, so the pick is tilted — whole profiles only — so that
`ptsd` lands on VA NCPTSD past-year by era, `depression` and `diabetes` on PLACES per ZIP,
and `substance_use_disorder` on NSDUH 2022–24 by era. Two Synthea medication artifacts are
removed when the profiles are distilled: insulin prescribed for prediabetes, and opioids
started more than 90 days before the reference date (CDC 2022's acute + subacute window). A sex × band pool with fewer than
`MIN_CARRIERS` carriers of a targeted flag borrows that flag's carriers from the nearest
bands (men 65–74 have none). `active_cancer_tx`, `suicide_risk` and `homeless` keep Synthea's
rates, have no cited target, and so carry a `_synthetic` flag. **What it costs:** depression keeps its PLACES gradient (per-ZIP r 0.42 before and
after); `active_cancer_tx` loses it (r 0.51 → 0.14), because PLACES measures ever-diagnosed
cancer and this flag is now in-treatment only.

**Acceptance:** `tests/unit/test_synthea_veterans.py` holds the four tilted flags to within 3
binomial standard errors of their cited rates, and the other three to within 3 SE of the
Synthea source rate reweighted to the cohort's own sex × band mix; it asserts that every
veteran's (flags, medication list) is one real profile's, that nobody is on insulin without
diabetes, and that opioid use sits at or below GAO's ~10% quarterly VA dispensing rate.

**Still genuinely synthetic** — no public source exists, so these keep parametric priors and a
`_synthetic` flag:

| field | how |
| --- | --- |
| deployment_era | by birth year; gulf/post911 share matches ACS era table |
| burn_pit_years (truth) | era = gulf/post911: LogNormal(mean 0.8 yr); else 0. Observed proxy: `pact_presumptive` (true positive 0.6, false positive 0.05) |
| ptsd_severity | ptsd=True → Categorical over 1–4; else 0 |
| home_ac | Bernoulli, rate declining across HVI band, correlated with `low_assets` |
| floor | basement 0.06, ground 0.25, upper 0.69 citywide; basement up-weighted ×2 where `stormwater_flooded_frac` is high |
| on_methadone_otp | 0.01 overall |
| caregiver type, given present | coresident / remote / va_pcafc split 0.55 / 0.35 / 0.10; `lives_alone=True` forces remote |
| med_rxcuis (the *assignment*) | the active list of the veteran's own Synthea profile (Track A, above). Every list is a real one; which veteran carries it is not. `diabetes` now comes from the same profile; `copd`, `asthma` and `chf` still come from PLACES. |
| mail_order_pharmacy | Bernoulli(0.80), from VA's published ~80 percent CMOP share. Synthea's FHIR export has no `dispenseRequest`, so this cannot be read. |
| days_supply_remaining | 90-day fill if mail order else 30-day; phase drawn uniform, so on any given day the cohort is spread across its refill cycle |

**Acceptance:** `test_cohort.py` asserts that for every PLACES-derived field, the cohort's
realised rate is within 20 percent of the ZIP-weighted source rate, and that every augmented
column has a `_synthetic` sibling set True.

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


---

## 6. Model

### 6.0 The ladder — build this way, not all at once

A hierarchical model with ICAR spatial effects, a latent exposure dose, distributed lags and
five correlated needs is a two-day research task. Build it as four rungs. **Every rung writes
the same `data/posterior.nc` contract**, so nothing downstream changes when you climb.

| Rung | Contains | Time | Note |
| --- | --- | --- | --- |
| **0 — prior-only** | No MCMC. Coefficients drawn from `priors.py` means; risk is a matrix multiply; intervals from prior draws. | 30 min | **Build first.** It is the demo's floor, not a fallback. |
| **1 — pooled NUTS** | Binomial cells, five needs, main effects, the 4-lag heat curve. No ICAR, no latent dose, no interactions. | 2 h | The realistic target |
| **2 — interactions + SiteDown** | The six interaction terms and ψ_k. This is where the story lives. | 2 h | Strong finish |
| **3 — ICAR + latent dose** | Spatial pooling, burn-pit measurement model. | stretch | Only if rung 2 merged early |

Report the rung you reached, its r-hat and what did not converge. A model that silently
failed to fit is worse than a simpler one that did.

### 6.1 design.py — shared by simulator and model
Builds `X_health (N×p)`, `X_int (N×q per hazard)`, `hazard tensors (Z×T×m)` and `lag stacks`: one row per veteran-day, columns in `design.FEATURES` order.

**The binomial cells moved to `hazard.cells()`** (rung 1, September 2026) and `design.py` was left alone, because the simulator shares it. Two changes to what this section originally specified, both in the safer direction:

- The grouping key is **the exact design row**, not `(zip, stratum, date)`. The likelihood depends on a veteran-day only through its row of `X`, so identical rows are one cell whatever ZIP or day they came from — which is *exactly* equal to the Bernoulli likelihood rather than approximately, and `tests/test_hazard_toy.py` asserts that against the panel. Measured over the 120-day panel: 1,200,000 veteran-days → **135,743 cells, 8.8×** (and the same 8.8× on the 6M Bernoulli terms — a cell still carries one binomial term per need, not one in total).
- Grouping is on the **active** columns for the rung, so at rung 1 two veterans who differ only in a medication interaction are one cell.

No `cells.parquet` is written: cells are derived from `cohort × hazards × site_status × outcomes` in about a second, and a cached copy is one more thing that can go stale against `truth.json`.

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
| θ heat × med_thermoreg_score | Normal(0.35, 0.3) per unit | CDC mechanism list; graded, so the prior is per score unit |
| θ heat × med_combo_raas_diuretic | Normal(0.5, 0.4) | CDC names this combination explicitly |
| θ heat × (acb_score ≥ 3) | Normal(0.5, 0.4) | ACB scale threshold |
| θ heat × med_renal_triple | Normal(0.6, 0.5) | NSAID + RAAS + diuretic, AKI with dehydration |
| θ outage × med_cold_chain | Normal(0.9, 0.6) | insulin spoils in about a day |
| θ maildisrupt × mail_order × supply_short | Normal(1.2, 0.6) | near-deterministic; wide anyway, and it should shrink |
| θ sitedown × med_controlled | Normal(0.9, 0.6) | retail emergency refill excludes controlled substances |
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


---

## 7. Decision layer

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
| switch_to_local_pickup | 0.10 | 0.05 | 0.05 | 0.65 | 0.10 | refill |
| pharmacist_med_review | 0.15 | 0.45 | 0.10 | 0.20 | 0.00 | pharmacist_slot |
| cold_chain_plan | 0.05 | 0.10 | 0.00 | 0.55 | 0.05 | call |
| controlled_substance_bridge | 0.00 | 0.00 | 0.20 | 0.70 | 0.20 | va_fill |
| check_in_call (Find out) | information action; value = expected reduction in epistemic variance × w | call |

### 7.3 eha.py
`EHA[i,a,t] = Σ_k w_k · mean_draws(p[i,k,t,draw]) · τ[a,k]`, plus `VOI[i] = Σ_k w_k · sqrt(epistemic_var[i,k,t])` for check-in calls.

### 7.4 allocate.py
Selection order is **(round, tier band, EHA per unit cost)**, under
`capacity = {call: 40, refill: 200, ride: 15, booking: 20, evac: 8, pharmacist_slot: 12, va_fill: 30}`;
the pharmacist slot is deliberately scarce, because it is a real person's afternoon; one action per veteran per day unless the veteran is Act-now, then up to 3. Optional `group_floor = {borough: 0.1}` for the fairness floor. Deterministic; returns `actions.parquet` with rank and rationale.

The three keys, and why they are in that order:

1. **Round.** Nobody is offered a second action until everyone has been offered a first. An Act-now veteran may hold three slots, and letting them take all three before a Find-out veteran gets one is monopoly rather than priority — and costly, because second and third actions are valued on the risk the first leaves behind, so they are worth a fraction of somebody else's untouched first. Filling scarce buckets that way costs ~17% of total EHA; taking the rounds in order costs ~6% and serves the same Act-now veterans.
2. **Tier band.** Within a round: Act-now, then Find-out, then Self-serve, then Everyday. Before it, allocation was pure greedy-by-EHA and a Self-serve veteran with broad moderate risk outranked an Act-now veteran with one sharp risk for the same call, so the tier badge promised a call the list never made. The band is a property of the **veteran on the work day** (the best tier among the risk days being worked for them that day), not of the individual candidate; banding candidates breaks capacity monotonicity, banding veterans does not, because within a veteran the order is still descending EHA.
3. **EHA per unit cost**, as before.

Together the first two keys are the promise the badge makes: *no Self-serve veteran holds a scarce-bucket slot while an Act-now veteran who could have used that bucket holds nothing at all.*

Raising capacity still never lowers total EHA, and `tests/unit/test_act_now_rules.py` pins both that and the 15% ceiling on what the band may cost.

### 7.4b Caregiver routing
If `caregiver != none` and `caregiver_contact_consent`, `care_team_call` and `verified_text` target the caregiver first (τ for those actions +0.10 on treatment_gap and access_loss, because a co-resident can act same-day). If `caregiver == none`, `verified_text` τ is halved and `care_team_call` / `evacuation_assist` are preferred; `assign_buddy` becomes available (τ access_loss 0.35, cost unit `partner_slot`). If `low_assets`, `cooling_center_ride` and `evacuation_assist` are booked, not suggested, and `heap_application` is added as a 5-day-out action (τ heat 0.30 over the season).

### 7.5 tiers.py
Act-now: p_mean ≥ 0.25 on any need with w ≥ 4 and epistemic share < 0.4, **or any row of the hazard-rule table matches** (below: site-dependent × SiteDown; `med_controlled` × SiteDown; `mail_order_pharmacy` and `days_supply_remaining ≤ 7` on a day the scenario disrupts delivery to that ZIP; no caregiver and `powered_equipment != none` in an outage). Find-out: epistemic share ≥ 0.4 and p_mean ≥ 0.10, **or the veteran's record has a gap that straddles a line the team acts on** — `p_gap_lo < L ≤ p_gap_hi` for L in (0.10, 0.25), where `p_gap_lo`/`p_gap_hi` (scores.parquet, from `model/score_prior.py`) are the p_mean this veteran would be reported with if the fields the VA does not have on file turned out to be their least- and most-risky values. Self-serve: p_mean 0.05–0.25. Everyday: rest.

The second Find-out rule is not a loosening of the first; it reads a different column, and that is why both are here. The epistemic share is Var(p) relative to p(1-p), and at rung 0 it is dominated by the prior spread every veteran shares rather than by anything about one of them: across 6,000,000 scored rows it reaches 0.426 against the 0.4 cut, so the first rule alone fires on 0.04% of veteran-days and the tier cannot be demonstrated. Lowering the cut is not the fix either, because Act-now requires the same share to be *below* it — at 0.25 Find-out reaches 3.1% and Act-now collapses from 1.37% to 0.004%. The two rules were competing for one threshold. Measured on the sandy_then_heat scenario, the gap rule brings Find-out to 2.4% of veteran-days (9.2% of actions) and leaves Act-now untouched at 16,424 veteran-days.

`assign()` returns `veteran_id, date, tier, tier_rule`, where `tier_rule` names the row that forced Act-now, or is null when the probabilities decided on their own. It takes `cohort`, `hazards` and `site_status` together, or none of them — a partial set is refused, because a half-applied rule table would silently stop some rules firing and still look like a full answer.

**The four hazard-triggered rules live in `leeward/decision/act_now.yaml`**, a clinician-editable table beside `severity.yaml` and `tau.yaml`, not in code. Each row ANDs up to three cells — `veteran` (cohort), `station` (site_status), `zip` (hazards) — and carries a `because` sentence that is printed verbatim in the action's rationale and on the veteran card, plus a `source`. A veteran who matches a row is Act-now **regardless of where their probability sits**, because each row encodes a mechanism the posterior has not seen. These are close to deterministic, which is the point: they are the rows a care team can act on with no argument.

| Rule | veteran | station | zip | Mechanism |
| --- | --- | --- | --- | --- |
| `site_dependent_at_a_closed_station` | dialysis, infusion or OTP | `site_down` | — | Station 630 is in evacuation zone 1 and its OTP closed five months after Sandy; ~100 veterans needed guest-dosing |
| `controlled_substance_at_a_closed_station` | `med_controlled` or OTP | `site_down` | — | The **VA Pharmacy Disaster Relief Plan**'s 10-day retail supply **excludes controlled substances**, so the fallback covering everyone else does not exist for them |
| `mail_order_supply_short_on_a_disrupted_day` | `mail_order_pharmacy` and `days_supply_remaining ≤ 7` | — | `mail_delivery_disrupted` | ~80% of VA outpatient prescriptions arrive by mail, so a flooded ZIP is a medication-supply event |
| `unattended_powered_equipment_in_an_outage` | `caregiver == none` and `powered_equipment != none` | — | `outage_frac ≥ 0.2` | emPOWER: the equipment stops and nobody in the home can act on it |

The two thresholds are editable numbers in the YAML, not constants: 7 days is the forecast window, and 0.2 matches `leeward.demo.OUTAGE_ALERT_FRAC`. `load()` validates every column name against `leeward/schema.py` and every operator against the column's type, so a typo fails loudly rather than quietly never firing again.

**Acceptance:** `test_allocate.py` hand-checkable 5-veteran case; capacity never exceeded; raising capacity never lowers total EHA. `test_act_now_rules.py` covers the four rules, the band, and the 15% EHA ceiling.


---

## 8. Outreach

- `messages.py`: templates per tier and hazard. Every message contains: channel tag (`VEText` / `MHV` / `care_team_phone`), a 4-word verification phrase from `verify.py` (deterministic per veteran-day from a seed; word list of 512 common words), the line *"The VA will never ask you to pay, wire money, or share bank details,"* `VSAFE 833-388-7233`, and `Veterans Crisis Line: dial 988, press 1`. Flood messages add the evacuation center with step-free access and a "pack list" (meds, equipment, chargers, IDs). Caregiver-addressed messages name the veteran, state the plan in second person to the caregiver, and never include diagnoses. Low-assets messages state what is free (HEAP, cooling centers, emergency refill voucher, VA transport) and never suggest a paid option.
- `export.py`: partner sheet CSV with `consent_partner_check, consent_ride, consent_housing` flags; rows without consent are excluded, never redacted.
- `outcome_log.py`: append-only parquet; `POST /log` writes it; `score.py` reads it to update per-veteran frailty (simple Beta-Binomial update on top of the cached posterior).

**Acceptance:** `test_messages.py` asserts every rendered message contains all five mandatory elements; no message contains a URL shortener or a phone number not in the allow-list.

---

## 9. API

| route | returns |
| --- | --- |
| `GET /region` | the served region: `region_id`, label, unit, subregions, `geojson_url` (relative), `geojson_property` |
| `GET /region/geojson` | that region's map base, read from its reference directory |
| `GET /forecast?scenario=&day=` | hazards for the next 7 days by ZIP (`geo_id`) and facility |
| `GET /scores?date=&need=` | ZIP and facility aggregates with intervals |
| `GET /veteran/{id}?date=` | card: p per need, interval, epistemic share, drivers, tier |
| `POST /actions` body `{date, capacity, group_floor?, prior_scale?, limit?}` | the work list for the **do-by day** `date` (each row carries `lead_days`; its risk day is `date + lead_days`) + total EHA + baselines' EHA + `n_too_late` + `n_not_reached` |
| `GET /message/{action_id}` | rendered message |
| `POST /log` | outcome log row |
| `GET /report` | eval JSON for the Model report screen |
| `GET /export?date=` | partner sheet CSV |


### 9.1 What the routes do beyond the table

`leeward/api/main.py` implements all eight. Every route reads cached parquets through
`schema.read`; only `POST /actions` computes, and it answers in about 130 ms at 10,000 veterans.

- **`day`** in `/forecast` is the offset from the scenario's first day; the window is seven days from there (shorter at the end). `facilities` is one `FacilityStatus` **per facility per day**, keyed like `site_status` itself, so a closure lands on the day it starts rather than over the whole window; a caller that wants "is this site there at all this week?" takes `.any()` over the window's rows.
- **`scenario`** and **`prior_scale`** are accepted only for what is cached: `sandy_then_heat` and `1.0`. Anything else is a 422 that says so; it is never answered with other data under the requested name.
- **`POST /actions`** merges a partial `capacity` onto `DEFAULT_CAPACITY` and echoes the merged dict. `counts_by_tier` counts action rows, not veterans. Unknown buckets, negative capacity and a bad `group_floor` are 422s. `date` is the day the team works, so the route reads the risk days that day can still act on (`date` .. `date` + the longest lead in `tau.yaml`) and returns what has to be done on `date`. Do-by days share no capacity, so this is still one request.
- **`n_not_reached`** is the other half of the slider, in the same answer: the Act-now and Find-out veterans a team with no capacity limit reaches with a person and this capacity reaches with nobody. A free `verified_text` is not being reached. It is one more greedy pass inside `compare(headroom=True)` over candidates the request has already valued -- about 5% of the request -- not a second allocation, so a board never has to ask twice.
- **`limit`** cuts the rows returned, highest EHA first, and nothing else: `n_selected`, `total_eha`, `counts_by_tier`, `baselines`, `n_too_late` and `n_not_reached` all describe the whole allocation, so a board rendering a top slice can say "40 of 6,303". It defaults to no limit, and every issued row is still resolvable by `GET /message`.
- **`baselines`** are the same team with the same capacity and the same candidate actions valued the same way, working the veterans in a different order: oldest first, most chronic conditions first, or a shuffle seeded by the date. Each veteran's own actions are still tried best first. So the gap to `leeward` is what risk-ranking who goes first is worth, not a strawman with fewer tools. Most of every total is the free `verified_text` bucket, which no ordering changes, so the gap is modest. Measured over every scored day (rung 0): 22-33% on the 500-veteran fixtures, and 0-17% (median 9%) on the 10,000-veteran run, where the days at 0% are the ones with no competition for a scarce slot. No baseline beat `leeward` on any day.
- **`GET /message/{action_id}`** accepts the `action_id` or the `msg-` `message_id`, and finds any action from a recent `POST /actions` as well as the cached plan. 503 until `leeward.outreach.messages` exists.
- **`POST /log`** is append-only, one row per `action_id`: a repeat is a 409. Unknown veteran is a 404.
- **`GET /report`** returns `report/report.json` as-is, failed fairness rows included. Before `make report` has run it returns only `model_rung` with `generated_at: null`, and an empty fairness table then means "not audited", not "passed".
- **`GET /export`** is `leeward.outreach.export.partner_sheet` on the cached plan: rows without consent are excluded, and the sheet carries no tier, risk or rationale.
- Any table that is missing is a 503 that names what to run; one that no longer matches its contract is a 500 that names the table. A file rewritten under a live server is picked up on the next request.

---

## 10. UI

Screens, in demo order: Forecast → Map → Care team list → Veteran card → Message → Model report.

- **Map:** deck.gl `GeoJsonLayer` of NYC ZIPs, fill = expected need count for selected need and day; `IconLayer` for VA facilities, red when SiteDown; toggle for evacuation zones and FloodNet sensor trips. Fallback: static Plotly choropleth.
- **Care team list:** table cut at capacity; tier badge; EHA; drivers as chips. **CapacitySlider** re-posts `/actions` on change, animates the harm-averted counter and the baseline bars.
- **Veteran card:** 5 needs as interval bars (10–90 pct) with epistemic share shaded; drivers in plain language; action plan; "Why this tier" line.
- **PriorSlider:** 0.5× / 1× / 2× prior scale; swaps between three cached posteriors.
- **Model report:** recovery dot-whisker, reliability curves, harm-averted bar chart vs baselines, ablation table, fairness table.

**Acceptance:** `make demo` boots API + UI in < 60 s from a clean clone; every screen renders with the stub API; slider round-trip < 300 ms.


---

## 11. Evaluation harness

| script | output | pass bar |
| --- | --- | --- |
| recovery.py | `report/recovery.json` (truth vs 90 pct interval per parameter) | ≥ 90 pct covered |
| calibration.py | reliability per need on days 91–120 | ECE < 0.03 |
| holdout.py | PR-AUC per need vs no-climate baseline | report both |
| ppc.py | daily counts by ZIP and by facility vs 90 pct band | ≥ 90 pct of days inside |
| sbc.py | 50 refits on 500-veteran draws from prior | rank histograms; runs overnight |
| ablate.py | drops climate / lags / interactions / latent dose / ICAR / SiteDown; calibration + harm averted per row | table |
| decision_quality.py | harm averted at K ∈ {20,40,80} for Leeward vs age / chronic-count / random | the Impact bar chart |
| fairness.py | ECE, FNR, reach and coverage-at-40-calls by borough, HVI band, evac zone, income band, caregiver status, medication burden, race/ethnicity | show gaps; flag > 20 pct relative FNR; report `direction` on the same bar applied to reach |
| report.py | assembles `report/report.json` for `GET /report` | |

**Why the fairness audit reports more than the flag.** The 20 pct relative FNR bar stays exactly
where it is, but on the real cohort it cannot be crossed: at a pooled FNR of 0.958, flagging a
group would take an FNR of 1.149. The held-out window holds ~16.7k veteran-days with a need and
40 calls a day to spend on them, so every group's FNR sits between 0.93 and 0.99 and two numbers
at the ceiling cannot diverge by 20 pct. A report that says only "0 of 33 flagged" has therefore
shown a metric with no failing state and called it a pass. So `fairness.py` also reports:

- **`reach`** = 1 − FNR, and `reach_ratio_to_cohort`. The identical measurement with the ceiling
  subtracted off, where groups separate by a factor of six rather than a tenth.
- **`direction`** ∈ `reached_more` / `reached_less` / `on_par` — which side of the *same* 20 pct
  bar the group's reach falls on. A group reached more than the cohort is a result, not the
  absence of one, and is marked as deliberately as a flagged row.
- **`coverage`** — the share of a group's events that got one of the `DEFAULT_CAPACITY["call"]`
  daily calls, from `leeward.decision.allocate` run over the same window. Null when the audit was
  given no call list: not measured and measured-as-nobody are different claims.

Reach and coverage disagree, and the disagreement is the finding. The tier rule surfaces
low-income (2.72×), no-caregiver (1.72×), HVI-5 (1.69×) and 5–9-medication (1.99×) veterans more
than the cohort; under a 40-call budget the allocator follows through on the first two (2.69× and
2.17×) but lands at parity on the other two (0.97× and 1.05×). `flag_is_reachable()` says which
regime a report is in, and every rendering of the table repeats it.


---

## 12. Definition of done (first version)

- `make demo` from a clean clone boots in < 60 s and runs offline, with no `.env` and no API key.
- The model rung actually fitted is written down, with r-hat.
- Both scenarios play end to end; Walter's card, the capacity slider and the SiteDown marker work.
- `report/report.json` shows recovery ≥ 90 pct, ECE < 0.03, harm averted vs baselines, ablation table, fairness table.
- Every message passes `test_messages.py`.
- `docs/sources.md` complete; two-pager, 90-s video and 8 slides in `docs/`.

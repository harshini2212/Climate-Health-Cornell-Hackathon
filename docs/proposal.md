# Leeward — Care that gets ahead of the weather

**Veteran care continuity under heat, flood, outage and smoke · Health in Climate AI Hackathon, NYC 2026 · Partner brief: McKesson / Cornell Tech**

*Proposal v3 · 19 Sep 2026 · Synthetic people, real places · no real patient data*

> **What changed in v3.** Every public source below was fetched, joined and verified on 19 Sep 2026 and is committed to this repo under `data/reference/`; see [`docs/sources.md`](sources.md) and [`data/README.md`](../data/README.md). Three claims that were previously assertions are now joins you can re-run, and four numbers have been corrected against primary sources. The cohort's neighbourhood rates are no longer invented: they come from CDC PLACES per ZCTA.

---

## 1. One-paragraph pitch

VA care is reactive: a veteran is seen after the ER visit, after the missed refill, after the flooded basement. Leeward is a Bayesian decision engine that reads the 3–7 day forecast (heat, coastal surge, flash flood, outage, smoke), scores every veteran on a care team's panel for five concrete needs, and returns **today's action list, cut at the team's real capacity**, with each call justified in plain language and sent through a channel a scammer cannot imitate. It does not replace the VA, HUD-VASH, Team Rubicon or the Veterans Crisis Line; it tells those systems who to reach first, two to five days before the event, and what to bring.

**What is new versus every other risk score:** (1) climate × exposure × treatment interactions from PACT Act and WTC Registry evidence, (2) uncertainty that *drives* triage rather than decorating it, (3) actions ranked by expected harm averted under staff capacity, (4) explicit coverage of the two challenge areas most teams skip: coordination and protection from exploitation, (5) a facility-disruption term, because in NYC the flood does not only hit the veteran; it has closed the VA itself, and (6) **the medication list as a climate exposure in its own right** — not "has chronic disease" but *which drugs, in what combination, with how many days left, arriving how*.

That sixth one is the cheapest large gain available, and almost nobody takes it. A risk score built on diagnoses says a veteran has hypertension. The prescription says he is on hydrochlorothiazide **and** lisinopril — the exact pairing CDC names as significantly increasing harm in heat — that he has nine days of supply left, and that those nine days are arriving by mail through a ZIP that floods. Diagnosis tells you he is vulnerable. The medication list tells you what will go wrong, on which day, and what to do about it on Tuesday.

---

## 2. The problem, in NYC terms

Four hazards, four different failure modes for continuity of care. A model built for heat alone misses three of them.

| Hazard | NYC evidence | What breaks for a veteran |
| --- | --- | --- |
| **Extreme heat** | ~500 premature deaths per warm season; ~490 heat-exacerbated per year (2014–2023) and 7 heat-stress per year (2016–2025), of which the June 2025 heat wave alone caused 19. Deaths occur overwhelmingly at home. Black New Yorkers die of heat stress at 3× the white rate, Latino New Yorkers at 2×. (NYC Health, 2026 Heat Mortality Report) | Heat-sensitive medications (diuretics, anticholinergics, beta-blockers, antipsychotics), COPD/CHF, living alone, and cooling that is owned but cannot be afforded to run |
| **Coastal surge** | Sandy (Oct 2012): the Manhattan VA Medical Center, which lies in a flood zone, evacuated ~100 patients before landfall and stayed closed for months; its dialysis unit and an opioid treatment program (~100 veterans on methadone) were displaced for five months; veterans travelled 90 minutes to Brooklyn for care | The **care site** closes, not just the home: dialysis, infusion, methadone dosing, pharmacy |
| **Pluvial flash flood** | Ida (1 Sep 2021): a record **3.15 in/hr** of rain against a sewer system designed for **1.75 in/hr**; NWS's first-ever flash-flood emergency for NYC; **11 of the city's 13 deaths were in basement apartments**, concentrated among Asian residents in Queens | Basement and ground-floor units, mobility impairment, wheelchairs, powered beds; no warning lead time; mold and respiratory harm afterward |
| **Power outage** (from any of the above) | Sandy took out backup generators at Bellevue and NYU; >3 million Medicare beneficiaries nationally rely on electricity-dependent equipment (HHS emPOWER) | Oxygen concentrators, ventilators, home dialysis, IV pumps, electric wheelchairs; telehealth and refrigerated insulin |
| **Wildfire smoke** | June 2023 Canadian smoke event, worst NYC air quality on record | PACT Act presumptive respiratory disease (asthma, COPD, bronchiolitis), cancer patients on treatment |

Layered on top: the PACT Act now presumes 20+ respiratory and cancer conditions are service-connected for Gulf War and post-9/11 veterans; the WTC Health Registry shows trauma plus airborne exposure produce co-occurring physical and mental illness that persists 20+ years. Older veterans carry all of this at once. **Nobody sees the layers together, so the need appears only after the harm.**

One detail from the 2026 heat report shapes the model directly. The rise in heat-exacerbated deaths is attributed mainly to more **"non-extreme hot days," 82 °F up to the 95 °F extreme-heat threshold** — not to more 95 °F days. That is why Leeward's heat term hinges at 82 °F rather than at an alert threshold. The hyperparameter is a citation, not a choice.

And the layers reach the care site itself. Joining the VHA facility registry to NYC's hurricane evacuation zones (`data/reference/va_facilities_nyc_hazard.parquet`) puts VA **station 630, the Margaret Cochran Corbin VA Campus in Manhattan, inside evacuation zone 1** — the first zone the city orders out. Station 630 is the campus that evacuated on 28 October 2012 ahead of Sandy, whose opioid treatment program then stayed shut for five months. That is not a historical anecdote we are citing; it is a two-second join we run on stage, and it is the entire case for the facility-disruption term.

---

## 3. Who we serve

Older and exposure-affected veterans in NYC, scored as a *combination*, never as a single group.

| Group | Why at risk | What the model watches |
| --- | --- | --- |
| Older veterans 65+ | Multimorbidity, polypharmacy, often alone | Heat illness, falls, missed refills, isolation |
| Burn-pit / airborne-hazard exposed | PACT Act presumptive respiratory disease and cancers | Flare-ups on smoke and post-flood mold days |
| PTSD or depression | Higher physical illness risk; disruption worsens symptoms | Missed therapy, crisis after displacement |
| Active cancer treatment, dialysis, serious lung/heart disease | No reserve; power- and site-dependent | Treatment gap, infection, equipment failure |
| **Flood-exposed housing** | Basement/ground-floor, evacuation zone 1–3, mobility impaired | Displacement, entrapment, loss of transport to care |
| **Medication-exposed** (new) | On drugs that impair thermoregulation, on the CDC-named additive combination, on a cold-chain drug, or on a controlled substance the emergency retail refill route excludes | Heat illness at a lower temperature than their neighbour; insulin spoiled by an outage; running out mid-event because the refill is in the mail |

---

## 4. Data

Real public data for places, hazards and facilities. A documented synthetic generator for people. Every real number is cited; every synthetic field carries a `_synthetic` flag.

The claim is deliberately narrow and we will say it in exactly these words: **every neighbourhood-level rate in the cohort is real and cited; only the people are synthetic.**

All of it is already in the repo. `scripts/fetch_sources.py` runs 18 keyless fetchers into `data/reference/` (~4 MB, committed) and records url, rows, bytes and a checksum for each in `manifest.json`. `make demo` reads only that directory, so the demo runs with the network off.

### 4.1 Hazard and context layers (real)

All verified live on 19 Sep 2026 and committed to `data/reference/`. **None of these needs an
API key.** Three that the first draft assumed were open are not any more, and the replacements
are noted; `docs/sources.md` §3 records every one so nobody re-derives it at 03:00.

| Layer | Feeds | Source, as actually fetched |
| --- | --- | --- |
| NWS alerts and 7-day gridpoint forecasts | Event trigger, lead time | api.weather.gov — keyless |
| **EPA AirNow daily monitor files** | PM2.5 covariate; the June 2023 smoke replay | files.airnowtech.org — keyless. *(airnowapi.org needs a key; the file service carries the same monitors.)* |
| **NYC Stormwater Flood Maps** (moderate 2.13 in/hr, current sea level) | Pluvial flood exposure per ZIP | NYC Open Data 9i7c-xyvv → `stormwater_by_modzcta.parquet` |
| **Hurricane Evacuation Zones 1–7** | Surge exposure per ZIP; facility exposure | NYC Open Data epne-qv9x, area-weighted onto MODZCTA |
| **FloodNet sensors and observed flood events** | Live flood detection; real replay events | NYC Open Data kb2e-tjy3 and aq7i-eu5q — *no data-request form needed, unlike floodnet.nyc itself* |
| FEMA National Risk Index, census tract | Long-run heat-wave, hurricane and coastal-flood priors | FEMA_NationalRiskIndex FeatureServer *(the old static zip now 301s away)* |
| **Heat Vulnerability Index 1–5** | ZIP heat prior; severity weights | NYC Open Data 4mhf-duep — *now published per ZCTA, so no NTA crosswalk* |
| **HHS emPOWER** | Powered-equipment prior; outage scenario | Public ArcGIS FeatureServer, layer 1, `STATE='NY'` → 1,702 ZIPs |
| **CDC PLACES 2025, per ZCTA** | The cohort's augment priors: mobility, social support, utility shutoff, transport, chronic disease | data.cdc.gov kee5-23sr |
| CDC SVI 2022, census tract | Vulnerability context and fairness strata | svi.cdc.gov |
| **ACS 2023 5-year B21001** | Veterans by ZCTA and age band → re-homing weights | Census Summary File on www2.census.gov *(api.census.gov now 302s to missing_key.html for every request)* |
| **VHA facility registry** | Where care is, and **which facility is itself in an evacuation zone** | VHA Medical Facilities FeatureServer *(api.va.gov needs an approved key; this mirror does not)* |
| NYC Parks Cool It! cooling sites | Destination set for the cooling-centre ride | NYC Open Data h2bn-gu9k |

Two sources from the working set are deliberately **not** load-bearing. PowerOutage.us is not
archived, so the outage term is driven by scenario rather than replay, and we say so. The
SAMHSA locator is a routing convenience, not a model input.

### 4.2 Synthetic veteran cohort

Base: the VA's public Synthea release (500+ clinical concepts, 90 disease modules, SDoH), catalogued as 10,000 synthetic veteran records.

> **Correction, and it matters for reproducibility.** The catalogue entry describes 10,000 records, but the single downloadable resource is a **4.0 GB zip named `csv_national_100k.zip`, root directory `csv_usa_100k/`, in Synthea CSV format** — roughly 100,000 records, not FHIR bundles. So the FHIR code path (`cohort/fhir_reader.py`, and the path that would point at a Lighthouse or Oracle Health sandbox) is developed and tested against Synthea's own **FHIR R4 sample**, 111 real bundles at 30 MB, and the 10,000-veteran cohort is drawn from the VA CSV release where the download completes and parametrically from ACS age bands where it does not. We will say which one the demo used.

We then:

1. **Re-home** each record to an NYC ZIP, sampled in proportion to ACS veteran counts by age band. ACS 2023 5-year table B21001 puts **131,195 veterans in NYC ZCTAs, 53.5 percent of them aged 65 or over**; the age bands sum exactly to the total, which is the check that the table was read correctly.
2. **Assign an NYC VA facility** (Manhattan, Brooklyn, Bronx, St. Albans, CBOCs) by nearest, so the facility-disruption term has a target.
3. **Augment** with fields Synthea lacks, drawing every neighbourhood rate from a real per-ZIP source rather than inventing it (see §4.4): deployment era and burn-pit dose; PTSD severity; home AC; powered equipment from emPOWER ZIP counts; active cancer treatment or dialysis; heat-sensitive medications flagged from the medication table; **housing floor (basement / ground / upper) and evacuation zone from the stormwater and surge maps**; mobility impairment; **caregiver status and household resources**.
4. **Simulate** 120 days of daily outcomes from a known generative model with fixed true coefficients, under two scripted scenarios. This planted truth is what the model must recover.
5. **Hide** 20 percent of AC, floor and deployment fields at random to create honest missingness.

### 4.3 Known gaps, stated as assumptions

- No veteran-level data is public; every person is synthetic and every individual effect is planted.
- Eligibility and enrollment are not public; the demo assumes all 10,000 are enrolled with a care team.
- Burn-pit dose and housing floor are unobserved in reality; both are latent variables with priors, not known numbers.
- Outage and flood-sensor feeds are not consistently archived; the outage and flash-flood scenarios are scripted from Sandy and Ida, not replayed from feeds.
- Intervention effects are literature priors, adjustable in the UI, not measured here.
- Individual income, assets and caregiver status are not public. The cohort now draws them from **CDC PLACES per-ZCTA measures** rather than from an assumed rate (§4.4). In a pilot, caregiver status comes from the VA record and the means-test field.

### 4.4 The rates are not invented any more

The first draft of this proposal assigned neighbourhood rates by assumption — mobility
impairment 0.18 for 65+, caregiver absent 0.35, AC access by HVI band. CDC PLACES publishes
all of them per ZCTA, so we use those instead. Measured across NYC's 178 MODZCTAs and grouped
by Heat Vulnerability Index band:

| HVI | ZIPs | Utility shutoff threat | Lacks emotional support | Mobility difficulty | Lacks transport | COPD |
| --- | --- | --- | --- | --- | --- | --- |
| 1 (lowest) | 34 | 5.3% | 24.8% | 9.3% | 6.0% | 3.9% |
| 3 | 33 | 8.9% | 30.8% | 13.2% | 10.1% | 5.3% |
| 5 (highest) | 37 | **17.2%** | **35.0%** | **18.8%** | **16.6%** | **6.7%** |

Every gradient is monotone in HVI, from a source that has never heard of the Heat
Vulnerability Index. Two of these columns do real work in the model. `shututility` — the share
of adults threatened with a utility shutoff — is the measured version of the mechanism NYC
Health names in its heat report: people died at home with an air conditioner they could not
afford to run. And `emotionspt` — lacks social and emotional support — is the measured version
of the first thing a clinician looks for. Both are per-ZIP numbers, not guesses.

What remains genuinely synthetic, because no public source exists: housing floor, burn-pit
years, PTSD severity, per-person AC, and every individual daily outcome. Each carries a
`_synthetic` flag and each is named on stage.

---

## 5. The Bayesian model

A hierarchical **discrete-time hazard model**. For veteran *i*, day *t* and need *k*, the probability the need arises that day. Five needs: breathing flare-up, heat illness, mental-health crisis, treatment or medication gap, and **loss of access** (displacement, transport or facility closure).

```
logit h[i,k,t] = α_k
               + φ[z(i),k]                        # ZIP effect, ICAR spatial prior
               + β_k · H_i                          # health status
               + γ_k · Λ_i                          # latent cumulative exposure dose
               + Σ_{l=0..3} δ[k,l] · f(Heat[z,t−l]) # distributed heat lag, hinge at 82°F
               + Σ_{l=0..2} ε[k,l] · PM25[z,t−l]    # smoke lag
               + ζ_k · Flood[z,t]                   # NWS flood alert × stormwater depth × evac zone
               + κ_k · Outage[z,t]
               + ψ_k · SiteDown[f(i),t]             # the veteran's own VA facility closed
               + θ_k · (Hazard[z,t] ⊗ V_i)          # interactions (see table)
               + η_k · U_i                          # past care use
               + σ_k · S_i                          # social support and resources: caregiver, income/assets, lives alone
               + ξ_i                                # shared frailty across the five needs
```

| Interaction (Hazard ⊗ V) | Example |
| --- | --- |
| Heat × heat-sensitive medication | Diuretic on a 95°F day |
| Smoke × PACT respiratory condition | COPD on a PM2.5 > 150 day |
| Outage × powered equipment | Oxygen concentrator during a Con Ed outage |
| **Flood × basement/ground floor** | Ida-style flash flood, ground-floor unit |
| **Flood × mobility impairment** | Wheelchair user in evacuation zone 1 |
| **SiteDown × site-dependent treatment** | Dialysis or methadone patient whose facility is closed (the Sandy case) |
| **No caregiver × any hazard** | Nobody to notice the missed dose, fetch the refill, or get them out; the effect is a *multiplier* on every other risk, not a separate cause |
| **Low income/assets × heat** | Has an AC unit but cannot afford to run it (the pattern in NYC heat-stress deaths) |
| **Heat × thermoregulatory medication load** | A graded score, not a flag: diuretic 1.0, antipsychotic 1.0, beta-blocker 0.8, antihistamine 0.6, summed across the list |
| **Heat × (ACE inhibitor or ARB) × diuretic** | The one combination CDC singles out by name as significantly increasing harm. 16 percent of the cohort carries it. |
| **Heat × anticholinergic burden ≥ 3** | Cannot sweat. ACB is a validated 0–3-per-drug scale; 3 or more is the accepted clinical threshold |
| **Heat × NSAID on top of a diuretic and a RAAS agent** | Acute kidney injury with dehydration |
| **Outage × cold-chain medication** | Insulin in a warm refrigerator is a treatment gap in about a day |
| **Mail delivery disrupted × mail-order pharmacy × days supply < lead time** | Four in five VA prescriptions arrive by mail. This term is nearly deterministic, and that is exactly what makes it actionable. |
| **SiteDown × controlled substance** | The VA emergency retail refill benefit **excludes** controlled substances — VA must fill them. A closed station is a hard stop, not an inconvenience. |
| **Low income/assets × flood or outage** | Cannot pay for a hotel, a cab, a generator, or to replace ruined medication and equipment |

**Seven design choices, each visible in the demo**

1. **Daily hazard with distributed lags.** Heat harms over 0–3 days, smoke over 0–2. Lag curves are constrained to be smooth (random-walk prior), so the model learns a shape from few parameters.
2. **Latent exposure with measurement error.** Deployment dose Λ_i has an era-based prior and a noisy indicator when a record carries a PACT Act presumptive diagnosis. Wide uncertainty in Λ flows into wide intervals, which route that veteran to a check-in call.
3. **Facility as a unit of risk.** SiteDown is a per-facility daily indicator driven by the facility's own evacuation-zone status and the surge forecast. This is the term no heat-only model has, and it is exactly what Sandy did to the Manhattan VA. The input table is a join, not an assumption: of the 14 NYC VA facilities, **station 630 (Manhattan) is in evacuation zone 1**, the Staten Island clinic is in zone 2, and Brooklyn VAMC is in zone 4. Three of them — Manhattan, Brooklyn and the Bronx — carry the site-dependent services (dialysis, infusion, opioid treatment) whose loss is a treatment gap rather than an inconvenience.
4. **Weakly informative priors, shown not hidden.** Centers from WTC Registry and PACT Act evidence, wide scales; a slider halves or doubles the prior scale and re-scores live. Rankings barely move. The heat hinge is the one hyperparameter that is not a judgement call: it sits at 82 °F because that is the threshold NYC Health's own mortality analysis identifies as the source of the increase.
5. **Social support as a multiplier, not a covariate.** Caregiver absence and low resources enter twice: as main effects on every need, and as interactions with every hazard. Two veterans with identical charts and the same forecast can differ two- to three-fold in risk depending on whether someone is in the apartment with them and whether they can afford to run the AC or pay for a cab. Clinicians told us this is the first thing they look for; the model treats it that way.
6. **The medication list is a covariate block, not a checkbox.** Synthea already emits an RxNorm code on every prescription, and NLM's RxNav resolves those codes into the **VA's own 576-class drug taxonomy** — so Leeward speaks the vocabulary the care team's clinical pharmacist already uses. `CV702` is LOOP DIURETICS to RxNav, to the VA formulary, and to us. From that we derive a thermoregulatory-risk score, an anticholinergic burden on the validated ACB scale, the CDC-named additive combination, cold-chain dependence, controlled-substance status, and days of supply remaining. The mechanisms are CDC's, cited per class in `data/reference/med_climate_risk.csv`, and a pharmacist can edit that file without touching code.

7. **Fit on sufficient statistics.** 10,000 × 120 × 5 Bernoulli rows collapse to binomial cells by ZIP × stratum × day; NumPyro NUTS on JAX fits in minutes on CPU; the posterior is cached and live scoring is a matrix multiply.

**Outputs.** Per veteran per day: posterior mean and 80 percent interval per need, epistemic share, top three drivers from posterior contributions (no black-box SHAP). Per ZIP and per facility per day: expected counts with intervals for staffing. Per care team: today's action list.

**Why not gradient boosting.** It would fit the synthetic data as well. It cannot carry a latent dose, cannot say how much of a prediction is ignorance, cannot borrow strength across small ZIPs, and cannot absorb 30 logged outcomes without a retrain. We say this on stage.

**What we will have actually fit.** The model is built as a ladder — prior-only, then pooled NUTS on binomial cells, then interactions and SiteDown, then the ICAR spatial prior and the latent dose — and every rung writes the same posterior contract. We will name the rung we reached, its r-hat, and what did not converge. A model that silently failed to fit is worse than a simpler one that did.

---

## 6. From risk to action: the decision layer

A prediction only matters if it changes what a care team does on Tuesday. For each veteran *i*, candidate action *a*, day *t*:

```
EHA[i,a,t] = Σ_k  w_k · E_posterior[ h[i,k,t] · τ[a,k] ]
```

*w_k* is a clinician-set severity weight (treatment gap for a dialysis patient ≫ missed wellness text); *τ[a,k]* is the fraction of need *k* that action *a* prevents, a literature prior shown and adjustable in the UI. Actions are chosen greedily under the team's capacity (calls/day, refills, rides, evacuation assists), a knapsack the team can re-run when staffing changes.

**Uncertainty as a feature.** The posterior splits into aleatoric (daily chance) and epistemic (things we do not know about this person). High epistemic width earns a cheap **Find out** call, because the expected value of that information is high. Narrow, high risk earns the expensive action.

| Tier | Trigger | Action | Owner |
| --- | --- | --- | --- |
| Act now | High hazard, narrow interval, site- or power-dependent, **or will run out of medication mid-event** | Call today; early refill; backup-power or evacuation plan; reroute dialysis/infusion; pharmacist medication review | VA care team |
| Find out | Wide interval, missing fields (floor, AC, deployment) | 3-minute check-in; re-score same day | Care team or volunteer partner |
| Self-serve | Medium hazard | Verified text: cooling/clean-air site, evacuation center, refill link, 988 press 1 | Automated via VA channels |
| Everyday | Low hazard | Monthly plan: shaded green space, movement, VA Whole Health | Automated |

**Action catalog by hazard and lead time**

| Need | 5 days out | 2 days out | During / after |
| --- | --- | --- | --- |
| Medication gap | Early refill via My HealtheVet, **targeted by days-supply remaining against the forecast window rather than sent to everyone** | **Switch mail-order to local window pickup** if delivery to that ZIP is at risk; confirm delivery | VA Emergency Pharmacy Program: any retail pharmacy, ≥10-day supply, with a VA bottle or script |
| **Controlled substance** | Flag early: the retail emergency refill route **excludes** controlled substances, so VA must fill them itself | VA fill before the event; for OTP, pre-dispense take-home doses where protocol allows | Guest-dosing arrangement — the Sandy playbook, ~100 veterans |
| **Heat-risk medication** | Route to the VA clinical pharmacist for a heat-interaction review. Leeward flags; the pharmacist decides. | Hydration and timing advice per protocol; storage guidance (nothing left in a hot car or a hot room) | Same-day pharmacist call if a heat alert lands while the review is open |
| **Cold-chain medication** | Confirm the insulin storage plan and a cooler before an outage window | Backup-power or ice plan; identify the nearest open pharmacy fridge | Replace spoiled supply; VA mobile pharmacy |
| Heat illness | AC check; HEAP cooling application if none | Cooling-center match, step-free transit | Wellness call day 2 of wave |
| Breathing | Inhaler/oxygen supply check | Clean-air room; windows-closed advisory | Same-day tele-visit; post-flood mold check |
| Treatment gap (dialysis, chemo, methadone, oxygen) | Register with utility medical-needs list; **pre-arrange alternate site** (the Sandy dialysis lesson) | Confirm ride or reschedule; **pre-dispense take-home doses where protocol allows** | Care-team call within 2 h of outage or site closure |
| **Loss of access / flood** | Evacuation-zone check; **basement/ground-floor flag**; evacuation-center match with accessibility | Transport booked; equipment and meds packed; buddy assigned | Displacement follow-up; temporary care-site assignment; HUD-VASH contact if housing lost |
| Mental health | Move therapy to phone if outage likely | Peer buddy check | Crisis signal → Veterans Crisis Line, 988 press 1 |
| **Caregiver present** | Message the caregiver as the primary contact (with consent); confirm the plan with them | Caregiver gets the pack list and evacuation-center match | Caregiver is the outcome-log contact |
| **No caregiver** | Assign a volunteer partner buddy (Team Rubicon, VFW post) 5 days out | Care-team call rather than text; transport booked, not suggested | Displacement follow-up prioritized |
| **Low income/assets** | HEAP cooling application; utility medical-needs list; confirm 90-day med supply | Pre-arranged transport and shelter, not "go to a cooling center" | Emergency Pharmacy Refill voucher; DAV / PVA / VFW disaster grants |

The tool proposes; clinicians approve every Act-now action. It never changes a medication, decides eligibility, or contacts a veteran outside VA-approved channels.

---

## 7. Coordination (area 04) and protection from exploitation (area 05)

**The pharmacist is a named owner, not an afterthought.** VA care teams already include clinical pharmacy specialists with their own scope of practice, and every medication action in Leeward routes to that person rather than to the physician's inbox or to an algorithm. Leeward flags a heat-interaction review; the pharmacist decides. **The tool never changes a dose, and the demo says so out loud.** CDC's own guidance tells clinicians to review medication lists for heat interactions and consider adjusting dose or fluid restriction on hot days — Leeward's contribution is telling them *which forty patients* to review before Thursday.

**One list, many hands.** The care team's action list exports, with consent flags, as a partner sheet: who opted in to a Team Rubicon wellness check, who needs a Combined Arms ride, who needs a HUD-VASH housing contact, who is on the utility medical-needs list. One owner (the VA care team); partners acknowledge back; the acknowledgment is an outcome-log row.

**Verified outreach.** After disasters, scammers pose as FEMA staff, adjusters and charities; VA runs a dedicated fraud line (VSAFE, 833-388-7233) for exactly this. Leeward's messages are built to be distinguishable from a scam:

- Sent only through VA channels the veteran already uses (VEText, My HealtheVet secure messaging, the care team's known number). Never a new number.
- Carry a 4-word verification phrase the veteran can read back to the care team.
- Always include: *the VA will never ask you to pay, wire money or share bank details*; the VSAFE number; 988 press 1.
- The Act-now packet includes a one-page "After the storm" scam card generated from VA and FTC guidance.

**Outcome log.** Each action records done / not done, reached / not reached, need occurred / did not. These 20–30 rows a week are what the Bayesian model absorbs without a retrain, which is how WTC-derived priors get tested instead of trusted.

---

## 8. EHR integration: Cerner / VistA ready from day one

Leeward's ingest reads **HL7 FHIR R4**, not CSVs. Synthea already emits FHIR R4, so the synthetic cohort and a real EHR share one code path.

| Leeward needs | FHIR resource |
| --- | --- |
| Demographics, address, floor (extension) | Patient |
| COPD, PTSD, cancer, PACT presumptives | Condition |
| Heat-sensitive meds, active chemo | MedicationRequest, MedicationAdministration |
| Oxygen concentrator, ventilator, wheelchair | Device, DeviceUseStatement |
| ER visits, missed appointments | Encounter, Appointment |
| Push an action to the team | Task |
| Log outreach | Communication, CommunicationRequest |
| Partner-export consent | Consent |

- **Oracle Health (Cerner) Millennium** exposes FHIR R4 with a public developer sandbox and SMART-on-FHIR launch, so the care-team screen can open inside the EHR. VA's Oracle Health rollout is live at 10 sites with 13 go-lives planned for 2026 and completion targeted for 2031; NYC sites are still on VistA.
- **VA Lighthouse Clinical Health API** is also FHIR R4 with SMART-on-FHIR, a sandbox and test users, and abstracts over VistA and Oracle Health. Building on FHIR makes Leeward Cerner-ready *and* VistA-ready with the same code.
- HL7 v2 (ADT/RDE feeds via an interface engine) is the event tap for the outcome log: an ADT A01 is an ER visit; a missing RDS is a refill that never dispensed.

Hackathon scope: FHIR ingest on Synthea bundles is in scope; a `--fhir-base` flag pointed at the Lighthouse or Oracle sandbox is a Sunday-morning stretch.

---

## 9. Validation: proof, not promises

Because the cohort comes from a generator with known coefficients, Leeward can show it recovers the truth. All checks render on one "Model report" screen.

| Check | Shows | Pass bar |
| --- | --- | --- |
| Parameter recovery | True coefficients vs 90 percent posterior intervals | ≥ 90 percent covered; dot-and-whisker plot |
| Calibration | Predicted 30 percent risks happen ~30 percent of the time, per need | ECE < 0.03; reliability curve |
| Held-out discrimination | Days 91–120 unseen | PR-AUC per need vs a no-climate baseline |
| Posterior predictive check | Daily counts by ZIP and by facility | Observed inside 90 percent band ≥ 90 percent of days |
| Simulation-based calibration | 50 small refits from prior draws | Flat rank histograms (overnight job) |
| **Decision quality** | Harm averted at K calls/day vs rank-by-age, rank-by-chronic-count, random | The single bar chart on the Impact slide |
| Ablations | Drop climate, lags, interactions, latent dose, spatial prior, **SiteDown** | Calibration and harm averted per row |
| Fairness audit | Calibration and false-negative rate by borough, HVI band, evacuation zone, income band, caregiver status, race/ethnicity | No group FNR > 20 percent relative above cohort; gaps shown, never hidden; capacity floor available |

What we will not claim: nothing here validates on real veterans. It validates that the machinery is right and the uncertainty is honest, which is the precondition for a VA pilot.

---

## 10. Demo and pitch

**Two scripted scenarios, both from NYC history**

- **Scenario A — Sandy-then-heat.** Day 1: NWS coastal flood warning, surge forecast for zones 1–2. Day 3: landfall, Con Ed outage in Lower Manhattan and the Rockaways, Manhattan VA marked SiteDown. Days 6–8: heat wave. Tests surge, outage, facility disruption and heat together.
- **Scenario B — Ida flash flood.** No lead time: NWS flash-flood emergency at 8 pm; FloodNet sensors trip in Queens; basement and ground-floor veterans in the ZIPs the stormwater map floods. The most exposed come out as the Rockaways (11692 is 59 percent flooded area), Coney Island and Lower Manhattan — which is the right answer, and a useful check that the join is real. Tests the model's behaviour when lead time is zero.

- **Scenario C — June 2023 smoke, replayed not scripted.** This one is not simulated at all. `airnow_pm25_nyc_smoke2023.parquet` holds what EPA's monitors actually recorded: PM2.5 peaks at **13.1 µg/m³ on 5 June, 101.0 on the 6th, 203.5 (AQI 254) at the Queens monitor on the 7th, 106.9 on the 8th, and 14.9 on the 9th.** Fifteen-fold in forty-eight hours and back. Tests the smoke lag against data a judge can check.

**The story.** "Walter" (synthetic): 71, Bronx, Gulf War era, COPD on the PACT list, oxygen concentrator, diuretic, ground-floor unit near the Bronx River, dialysis three times a week at the Manhattan VA. Monday: forecast says surge Wednesday, heat Friday. Today he is one of 1,200 on his care team's panel.

**Live sequence (2 minutes)**

1. Forecast panel: NWS, AirNow, FloodNet; 10,000 veterans re-scored in under a second.
2. Map: expected needs by ZIP and by facility for 7 days; Lower Manhattan and the Rockaways light up for surge, the Bronx for heat; the Manhattan VA marker turns red.
3. Care-team list cut at 40 calls. Walter is #2: 71 percent (58–84) chance of a treatment gap by Thursday. Drivers: dialysis × SiteDown, oxygen × outage, ground floor × flood. Actions: call today; book dialysis at Brooklyn for Wednesday; backup-power plan; transport Friday to a cooling center.
4. Capacity slider 40 → 20 → 80: the list re-ranks, the harm-averted counter moves, the Find-out tier fills.
5. The message Walter receives: VA channel, verification phrase, VSAFE, 988 press 1, never-pay line.
6. Model report: recovery, calibration, harm averted vs baselines, fairness by borough and evacuation zone.

**Pitch (3 minutes)**

- 0:00 One number and one date: ~500 heat deaths a year; and 28 Oct 2012, the night the Manhattan VA evacuated.
- 0:30 Walter, and what usually happens: nobody calls until Friday's ER visit.
- 1:00 Three questions: who, when, what changes if we act.
- 1:30 Why Bayesian, plainly: starts from what PACT and the WTC Registry taught us, says how sure it is, and uncertainty decides who gets a 3-minute call vs a refill and a ride.
- 2:00 Proof: recovers planted truth, calibrated, averts N× the harm of calling the oldest first at 40 calls a day.
- 2:30 Areas 04 and 05: one list, many hands; messages a scammer cannot copy.
- 2:50 The ask: one summer, one NYC VA care team, priors validated on de-identified data under IRB.

**Rules.** One talks, one drives. One number or one face per slide. Say "synthetic" three times. Name the WTC-prior limit before judges do. Rehearse the slider.

**Submission pack.** This document as a two-pager; 90-second video; public repo with one-command `make demo`; 8 slides.

---

## 11. Risks and limits

| Risk | Mitigation |
| --- | --- |
| NUTS divergences on the latent-dose block | Non-centered parameterization; fallback fixes Λ at prior mean, stated openly |
| Feed outage during demo | All feeds snapshotted Saturday; demo runs offline |
| Synthea VA release lacks a field | Augment step owns every added field; Synthea supplies only demographics, conditions, meds, encounters |
| The VA Synthea resource is 4 GB of CSV, not FHIR bundles | FHIR path runs on Synthea's 30 MB FHIR R4 sample; cohort falls back to ACS-parametric demographics. Stated openly, both paths committed. |
| An upstream endpoint moves or starts wanting a key mid-event | Every source is already snapshotted into `data/reference/` and committed. `make demo` never touches the network. |
| "Bayesian" heard as slow or academic | Lead with Walter and the slider; model report last |
| Fairness audit fails | Show it; apply capacity floor; it reads as rigor |

Stated limits: all veterans synthetic; WTC/PACT evidence sets prior centers only and rankings are shown robust to doubling or halving them; τ are literature priors; clinicians approve every Act-now action.

## 12. After the hackathon

1. Validate priors on de-identified VA data with a VA research partner under IRB.
2. Pilot one summer with one NYC VA medical center care team; metric: harm averted at fixed staff time vs the prior summer.
3. Wire SiteDown to VA Office of Emergency Management facility status so facility closures feed the model automatically.
4. Connect the everyday tier to VA Whole Health and NYC Parks shade data.
5. Open-source the cohort generator and evaluation harness as a shared NYC benchmark.

---

## Sources

Full citations, every fetched endpoint, and a table of the endpoints that moved are in
[`docs/sources.md`](sources.md). The short list:

- Health in Climate AI 2025 Devpost rubric — https://health-in-climate-ai-hackathon.devpost.com/
- NYC Health, 2026 Heat Mortality Report — https://a816-dohbesp.nyc.gov/IndicatorPublic/data-features/heat-report/
- NYC Health, Heat Vulnerability Index — https://a816-dohbesp.nyc.gov/IndicatorPublic/data-features/hvi/
- NYC Health, What Ida and Sandy taught us about flooding and health — https://a816-dohbesp.nyc.gov/IndicatorPublic/data-stories/flooding-and-health/
- NYC Open Data, Stormwater Flood Maps — https://data.cityofnewyork.us/Environment/NYC-Stormwater-Flood-Maps/9i7c-xyvv
- FloodNet NYC, sensors and data — https://www.floodnet.nyc/methodology
- Lukowsky et al., Access to care for VA dialysis patients during Superstorm Sandy — https://pmc.ncbi.nlm.nih.gov/articles/PMC6661787/
- Griffin et al., Extended closure of an opioid treatment program after Hurricane Sandy — https://doi.org/10.1177/0022042618779541
- American Legion, VA updates on Manhattan campus evacuation — https://www.legion.org/information-center/news/veterans-healthcare/2012/november/va-updates-legion-on-manhattan-facility
- HHS emPOWER public REST service — https://services2.arcgis.com/ZQ4jTQn6k7VPXEwO/arcgis/rest/services/HHS_emPOWER_REST_Service_Public/FeatureServer
- VA Synthea synthetic veteran dataset — https://catalog.data.gov/dataset/synthetic-suicide-prevention-dataset-with-sdoh (4.0 GB CSV release; see `data/README.md`)
- Synthea FHIR R4 sample data — https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_fhir_latest.zip
- CDC PLACES 2025, ZCTA — https://data.cdc.gov/d/kee5-23sr
- NYC Hurricane Evacuation Zones — https://data.cityofnewyork.us/d/epne-qv9x
- FloodNet on NYC Open Data — https://data.cityofnewyork.us/d/aq7i-eu5q
- EPA AirNow daily files — https://files.airnowtech.org/airnow/
- VHA Medical Facilities (keyless) — https://services2.arcgis.com/VFLAJVozK0rtzQmT/arcgis/rest/services/Veterans_Health_Administration_Medical_Facilities/FeatureServer
- VA, PACT Act and your benefits — https://www.va.gov/resources/the-pact-act-and-your-va-benefits/
- VA, Disaster help and fraud line — https://www.va.gov/resources/disaster-help/
- VA News, Natural-disaster fraud prevention (Aug 2026) — https://news.va.gov/148481/standing-strong-after-the-storm-natural-disasters-fraud-prevention/
- VA Lighthouse Clinical Health API (FHIR R4) — https://developer.va.gov/explore/api/clinical-health/test-users
- Oracle Health Millennium FHIR R4 APIs — https://docs.oracle.com/en/industries/health/millennium-platform-apis/index.html
- Federal News Network, VA EHR rollout resumes (Apr 2026) — https://federalnewsnetwork.com/it-modernization/2026/04/va-ehr-rollout-resumes-after-three-year-pause/
- Hackathon problem page — https://healthinclimate.ai/hackathons/nyc/2026/problems/veteran-care-continuity  
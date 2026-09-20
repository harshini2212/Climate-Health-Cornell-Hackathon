# Leeward data

Two directories, two very different promises.

| | `data/reference/` | `data/raw/` |
| --- | --- | --- |
| Committed to git | **Yes** | No (gitignored) |
| Size | ~4 MB total | ~550 MB, plus a 4 GB optional |
| Contents | Small, joined, analysis-ready tables | Big upstream downloads and extracts |
| Promise | `make demo` reads only this, and works with the network off | Rebuildable, never required at demo time |

Everything in `reference/` was produced by `scripts/fetch_sources.py`, which records the
source URL, row count, byte size and a sha256 prefix for each file in
`reference/manifest.json`. Full citations with URLs are in [`docs/sources.md`](../docs/sources.md).

```bash
python scripts/fetch_sources.py --list          # what exists
python scripts/fetch_sources.py                 # refresh the small, keyless sources
python scripts/fetch_sources.py --only hvi evac # refresh a couple
python scripts/fetch_sources.py --heavy         # + stormwater, ACS summary file, Synthea
```

A fetcher that fails prints a warning and leaves the existing snapshot in place. Nothing in
the build is allowed to depend on the network at demo time.

---

## The geography key

Everything joins on **`modzcta`** — NYC's Modified ZIP Code Tabulation Area, 178 of them.
Not raw ZIP, not ZCTA, not NTA. NYC publishes it, the Health Department reports on it, and
`nyc_modzcta.geojson` is the map base the UI draws.

Where a source is published per ZCTA5 (CDC PLACES, HVI, ACS) the codes coincide for almost
every NYC ZIP; `nyc_modzcta.parquet` carries a `zcta_members` column listing the ZCTAs each
MODZCTA absorbs, for the handful that differ. Where a source is per census tract (SVI, FEMA
NRI), aggregate to MODZCTA yourself — those two are context layers, not join keys.

---

## `data/reference/` — the committed tables

### Geography and map base

| File | Rows | What it is |
| --- | --- | --- |
| `nyc_modzcta.parquet` | 178 | MODZCTA code, label, member ZCTAs, population estimate, centroid lon/lat, area |
| `nyc_modzcta.geojson` | 178 | The polygons. This is what deck.gl renders. |

### Hazard exposure per ZIP

| File | Rows | What it is |
| --- | --- | --- |
| `hvi_by_zcta.parquet` | 184 | NYC Heat Vulnerability Index, 1 (lowest) to 5 (highest), per ZCTA20 |
| `evac_zone_by_modzcta.parquet` | 178 | Hurricane evacuation zones 1–7 area-weighted onto MODZCTA. `evac_zone_min` is the most urgent zone covering ≥2% of the ZIP; `evac_frac_z1…z7` are the area fractions. |
| `stormwater_by_modzcta.parquet` | 178 | Share of each ZIP inside the NYC Stormwater Flood Map moderate-rain (2.13 in/hr) current-sea-level extent. The pluvial / Ida term. |
| `fema_nri_nyc_tract.parquet` | 2,324 | FEMA National Risk Index per census tract: overall risk, social vulnerability, community resilience, and the heat-wave (`HWAV_*`), hurricane (`HRCN_*`) and coastal-flood (`CFLD_*`) families |

Sanity check on the stormwater layer — the most-flooded ZIPs come out as the Rockaways
(11692 at 59%, 11693, 11694, 11697), Coney Island (11224) and Lower Manhattan (10004). That
is the right answer, which is a useful thing to know before you trust a join.

### People and need, per ZIP

| File | Rows | What it is |
| --- | --- | --- |
| `acs_veterans_by_zcta.parquet` | 212 | ACS 2023 5-year table B21001: veterans per ZCTA by age band (18–34, 35–54, 55–64, 65–74, 75+). **131,195 NYC veterans, 53.5% aged 65 or over.** Age bands sum exactly to the total. Only summary level 860 (ZCTA5) rows are kept; an earlier filter also caught census tracts whose ids end in an NYC ZIP. This is the re-homing weight. `pop_65plus` (all residents 65+) is the denominator that turns emPOWER counts into a per-ZIP rate. |
| `acs_race_by_zcta.parquet` | 212 | ACS 2023 5-year table B03002 (Hispanic or Latino origin by race), **all residents**, per ZCTA: 14 race-by-origin cells plus totals, 8.58 M people in NYC (28.3% Hispanic, 31.1% non-Hispanic white, 20.9% non-Hispanic Black, 14.7% non-Hispanic Asian). Cells add to the total on every row (asserted in the fetcher and in `tests/test_reference.py`). This is where the cohort's `race` and `ethnicity` come from. CDC SVI's `EP_MINRTY` was the obvious candidate and is not enough: it is one composite (everyone but non-Hispanic white), so it cannot say Black from Asian from Hispanic, and SVI is per tract with no tract→MODZCTA crosswalk in the repo. It is used instead as an independent cross-check by borough. **Not veterans:** no public table gives veterans' race per ZIP. |
| `places_zcta_nyc.parquet` | 186 | CDC PLACES 2025 per ZCTA. 22 measures. **This is where the cohort's augment priors come from.** See below. |
| `empower_ny_zip.parquet` | 1,702 | HHS emPOWER, electricity-dependent Medicare beneficiaries per NY ZIP, split by device family. NYC totals: **36,146 power-dependent, 3,165 on oxygen, 1,948 facility ESRD dialysis.** |
| `svi_nyc_tract.parquet` | 2,324 | CDC/ATSDR Social Vulnerability Index 2022, NYC tracts. Context layer and fairness strata. |

### Medication

| File | Rows | What it is |
| --- | --- | --- |
| `med_climate_risk.csv` | 52 | **VA drug class → climate mechanism.** Hand-curated from CDC's clinician guidance on heat and medications. Columns: `mechanism`, `hazard`, `weight`, `acb` (anticholinergic burden 0–3), `controlled`, `cold_chain`, `narrow_ti`. Pharmacist-editable, like `severity.py` and `tau.py`. |
| `va_drug_class_members.parquet` | 4,222 | **RxNorm code → VA drug class**, pulled from RxNav for every class in the crosswalk. Lets the cohort map Synthea prescriptions offline, with no RxNav call at demo time. |
| `synthea_med_profiles.parquet` | 109 | **One row per Synthea bundle: that patient's active RxNorm codes**, sex, and age at their last recorded event. Distilled from the 30 MB FHIR sample by `profiles_from_fhir()`, so a clean clone gets real medication lists — co-prescribing intact — without `data/raw/`. 77 of the 109 carry at least one active medication. |

RxNav publishes the **VA's own 576-class drug taxonomy**, keyless — which means Leeward
speaks the vocabulary a VA clinical pharmacist already uses. `CV702` is LOOP DIURETICS to
RxNav, to the VA formulary, and to us.

Synthea emits RxNorm codes on every `MedicationRequest`, so the prescription layer is **real
data the cohort was already carrying and the first draft ignored**. Measured on the 109-bundle
Synthea FHIR sample:

| | Share of the 77 patients on active meds |
| --- | --- |
| On ≥1 medication in the crosswalk, of any hazard | **77%** |
| On ≥1 medication that impairs heat response (`med_thermoreg_score > 0`) | **65%** |
| On **ACE inhibitor or ARB + a diuretic** — the combination CDC names as additive heat risk | **16%** |
| On a controlled substance (cannot use the retail emergency refill route) | 10% |
| On a cold-chain medication (insulin) | 9% |
| On a narrow-therapeutic-index medication | 8% |
| Anticholinergic burden ≥ 3 (clinically meaningful on the ACB scale) | 12% |
| Median active medications | 3 (max 14) |

An earlier version of this table put **77%** on the heat row and **5%** on the ACB row. 77%
is the share on *any* crosswalk medication, whatever its hazard; heat alone is 65%. ACB is
12% under the most-specific-class rule in `cohort/medications.py`. Every figure above is
asserted in `tests/test_medications.py`, which re-derives them straight from the bundles
whenever `data/raw/` is present — so the table cannot drift from the data again.

Those numbers come from a general-population sample. The veteran 65+ cohort runs higher, and
does: 72% heat-impairing, 18% on the CDC pair, 11% cold-chain, 4.4 active medications each.

The five most-prescribed drugs in the sample are insulin, hydrochlorothiazide, lisinopril,
metformin and amlodipine — so the medication terms fire on the *ordinary* patient, not an
exotic one. Hydrochlorothiazide plus lisinopril is exactly the pair CDC singles out.

**What is real here:** the RxNorm codes (from the record), the VA class mapping (RxNav), and
the mechanism and combination rules (CDC). **What is synthetic:** `days_supply_remaining` and
`mail_order_pharmacy`, because Synthea's FHIR export carries no `dispenseRequest` block. Both
are drawn from VA's published conventions — 30-day window fills, 90-day mail fills, and the
~80% of VA outpatient prescriptions that go by mail — and both carry a `_synthetic` flag.

Also synthetic: **which** of those 109 real lists a given synthetic veteran carries. The
cohort bootstraps whole lists from `synthea_med_profiles.parquet`, stratified by age band
only (18–54 / 55+), because the sample's medication burden triples at 55+ — 1.9 active meds
below, 5.4 above — and our panel is 54% over 65. Whole lists, never drug-by-drug, so real
co-prescribing survives. The draw ignores the veteran's own diagnosis list, so a cold-chain
medication does not imply the diabetes flag; the sample has six diabetics, too few to
condition on without inventing the structure. That closes with the Synthea swap.

### Care sites and hazard to those sites

| File | Rows | What it is |
| --- | --- | --- |
| `va_facilities_ny.parquet` | 84 | Every VHA facility in New York State: station number, name, address, lat/lon |
| `va_facilities_nyc_hazard.parquet` | 14 | **The SiteDown input table.** The 14 NYC VA facilities joined to their hurricane evacuation zone. |
| `nyc_cooling_sites.parquet` | 271 | NYC Parks Cool It! cooling sites — the destination set for the cooling-centre ride action |

The facility table is the one to read before you write the pitch:

| Station | Facility | Evac zone |
| --- | --- | --- |
| **630** | **Margaret Cochran Corbin VA Campus (Manhattan)** | **1** |
| 630GB | Staten Island Community VA Clinic | 2 |
| 630A4 | Brooklyn VA Medical Center | 4 |
| 630A5 | St. Albans VA Medical Center | 6 |
| 526 | James J. Peters VAMC (Bronx) | 0 |

Station 630 is the campus that evacuated on 28 October 2012 ahead of Sandy. It is in the
first zone New York City orders to evacuate. That join is the argument for the SiteDown
term, and it runs in about two seconds.

### Observed hazard, for replay scenarios

| File | Rows | What it is |
| --- | --- | --- |
| `airnow_pm25_nyc_smoke2023.parquet` | 124 | EPA AirNow monitor PM2.5 for 5–11 June 2023, NYC metro box. The Canadian-wildfire smoke episode. |
| `floodnet_sensors.parquet` | 491 | FloodNet sensor deployments: id, street, borough, ZIP, lat/lon, install date |
| `floodnet_events.parquet` | 3,269 | Observed street-flooding events: start, end, max depth in inches, onset and drain time, duration above 4/12/24 inches |
| `nws_forecast_nyc.parquet` | 70 | A snapshot of the NWS 7-day forecast for the five borough grid points |
| `nws_alerts_ny.json` | — | A snapshot of NWS active alerts for New York State |

The smoke file is not a simulation. It is what the monitors recorded:

| Date | Peak PM2.5 µg/m³ | Peak AQI |
| --- | --- | --- |
| 5 Jun 2023 | 13.1 | 53 |
| 6 Jun 2023 | 101.0 | 175 |
| **7 Jun 2023** | **203.5** (Queens) | **254** |
| 8 Jun 2023 | 106.9 | 178 |
| 9 Jun 2023 | 14.9 | 57 |

Re-run `--only nws_snapshot` on the morning of the demo so the forecast panel shows
something current, then do not touch the network again.

---

## The augment priors — read this before writing `cohort/augment.py`

The original spec invented most of the cohort's neighbourhood rates. It does not need to.
CDC PLACES publishes them per ZCTA, and they are already in the repo. Measured across NYC's
178 MODZCTAs, grouped by Heat Vulnerability Index band:

| HVI | ZIPs | Utility shutoff threat | Lacks emotional support | Mobility difficulty | Lacks transport | COPD | Independent-living difficulty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 34 | 5.3% | 24.8% | 9.3% | 6.0% | 3.9% | 5.5% |
| 2 | 36 | 6.3% | 26.6% | 10.6% | 7.3% | 4.3% | 6.2% |
| 3 | 33 | 8.9% | 30.8% | 13.2% | 10.1% | 5.3% | 7.7% |
| 4 | 37 | 11.9% | 33.2% | 15.2% | 12.8% | 5.8% | 9.2% |
| 5 | 37 | **17.2%** | **35.0%** | **18.8%** | **16.6%** | **6.7%** | **11.4%** |

Every gradient is monotone in HVI, from a source that has never heard of the Heat
Vulnerability Index. Map the cohort fields onto these columns rather than inventing rates:

| cohort field | draw from | note |
| --- | --- | --- |
| `low_assets` | `shututility_crudeprev` | The measured version of "owns an AC, cannot afford to run it" |
| `caregiver == none` | `emotionspt_crudeprev`, `loneliness_crudeprev` | Lacks social and emotional support |
| `mobility_impaired` | `mobility_crudeprev` | |
| transport barrier | `lacktrpt_crudeprev` | Gates whether a ride is suggested or booked |
| `copd`, `asthma`, `active_cancer_tx`, `depression` | `copd_`, `casthma_`, `cancer_`, `depression_crudeprev` | |
| `powered_equipment` | `empower_ny_zip.parquet` ÷ ACS 65+ | Per-ZIP rate, capped |
| `race`, `ethnicity` | `acs_race_by_zcta.parquet` (ACS B03002) | One joint draw per veteran from their own ZIP's cells. The ZIP's all-ages population, not its veterans, so it likely overstates diversity among the oldest. `_synthetic`. |
| ZIP sampling weights | `acs_veterans_by_zcta.parquet` | `P(zip | age_band) ∝ vet_<band>` |

**Still genuinely synthetic**, because no public source exists: `floor`
(basement/ground/upper), `burn_pit_years`, `ptsd_severity`, `home_ac` at the person level,
and every individual outcome. Flag all of them `_synthetic` and say so on stage. The claim
to make is precise and defensible: *every neighbourhood-level rate is real and cited; only
the people are synthetic.*

---

## `data/raw/` — rebuildable, never committed

| Path | Size | Why it exists |
| --- | --- | --- |
| `synthea_sample_fhir.zip` | 30 MB | 111 Synthea FHIR R4 bundles. The fixture the FHIR reader and its tests run against. |
| `synthea/va_synthea_csv_national_100k.zip` | 4.0 GB | The VA public Synthea release. See the warning below. |
| `nyc_stormwater_flood_maps.zip` + `stormwater/` | 82 MB | The GIS source behind `stormwater_by_modzcta.parquet` |
| `acsdt5y2023-b21001.dat`, `acsdt5y2023-b03002.dat`, `acs2023_geos.txt` | 340 MB | ACS Summary File sources behind `acs_veterans_by_zcta.parquet` and `acs_race_by_zcta.parquet` |

### A warning about the VA Synthea release

The dataset is catalogued as *"Synthetic Suicide Prevention Dataset with SDoH — 10,000
synthetic Veteran patient records generated by Synthea"*, and the proposal originally
assumed it shipped 10,000 FHIR R4 bundles. It does not. The single resource is a **4.0 GB
zip named `csv_national_100k.zip` whose root directory is `csv_usa_100k/`** — Synthea **CSV**
output, on the order of 100,000 records. Over a normal conference connection that is a
multi-hour download.

Plan accordingly:

- The **FHIR code path** — `cohort/fhir_reader.py` and its tests — runs against
  `synthea_sample_fhir.zip`. It is 30 MB, it is real FHIR R4, and it is enough to prove the
  same reader would point at a Lighthouse or Oracle Health sandbox.
- The **10,000-veteran cohort** uses the VA CSV release if the download has finished, and
  parametric demographics drawn from the ACS age bands if it has not. Either way the
  neighbourhood priors above are unchanged, because they do not come from Synthea.
- Say which one you used. "We used Synthea's FHIR sample for the reader and drew the cohort
  parametrically because the VA release is a 4 GB CSV" is a fine answer. Claiming FHIR
  bundles you did not read is not.

---

## API keys: none required

Three upstream APIs the spec originally named now need credentials. All three have keyless
equivalents, and the build uses those, so a clean clone with no `.env` produces a working
demo.

| Wants a key | What broke | What the build uses instead |
| --- | --- | --- |
| `api.census.gov` | Every request 302s to `missing_key.html` | ACS Summary File on `www2.census.gov` — keyless |
| `airnowapi.org` | `{"WebServiceError":[{"Message":"Invalid API key"}]}` | `files.airnowtech.org` daily files — keyless, same monitors |
| `api.va.gov/services/va_facilities` | `401 No API key found in request` | VHA Medical Facilities ArcGIS FeatureServer — keyless |

One more that moved: FEMA's National Risk Index static download at
`hazards.fema.gov/nri/Content/StaticDocuments/...` now 301-redirects to a landing page. The
live `FEMA_NationalRiskIndex` FeatureServer is the working path and is what the fetcher uses.

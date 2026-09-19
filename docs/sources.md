# Sources

Every real number that appears in the UI, the deck or either document is here with a URL.
Numbers that are not here are synthetic, and the code that produces them sets a
`_synthetic` flag.

Verified against the live endpoints on **19 September 2026**. Where an endpoint had moved or
started requiring a key, both the broken path and the working replacement are recorded, so
nobody re-discovers it at 03:00.

---

## 1. Numbers we quote

### Heat

> NYC Health, *2026 NYC Heat-Related Mortality Report* —
> https://a816-dohbesp.nyc.gov/IndicatorPublic/data-features/heat-report/

- About **500** New Yorkers die prematurely each summer because of hot weather.
- **~490 heat-exacerbated deaths per year**, 2014–2023 (heat worsening an underlying illness).
- **7 heat-stress deaths per year** on average, 2016–2025; 71 in total over those ten years.
- The June 2025 heat wave (22–25 June) produced **19 heat-stress deaths**, of 21 that year.
  Heat index peaked at **103 °F at LaGuardia** on 24 June; Central Park hit 98 °F, breaking an
  1888 record.
- Heat-related deaths are about **3% of all warm-season (May–September) deaths**.
- The increase is attributed mainly to more **"non-extreme hot days," 82 °F up to the 95 °F
  extreme-heat threshold**. *This is the source of the model's 82 °F heat hinge.*
- **Black New Yorkers die of heat stress at three times** the rate of white New Yorkers;
  **Latino New Yorkers at twice** the rate. This is why the fairness audit is a screen.
- Deaths overwhelmingly occur **at home**, and lack of home cooling is the dominant risk
  factor — including cooling people own but cannot afford to run.

> NYC Health, Heat Vulnerability Index —
> https://a816-dohbesp.nyc.gov/IndicatorPublic/data-features/hvi/

### Flooding

> NYC Health, *What Hurricane Ida and Superstorm Sandy taught us about flooding and health* —
> https://a816-dohbesp.nyc.gov/IndicatorPublic/data-stories/flooding-and-health/

- Hurricane Ida, 1 September 2021: a record **3.15 inches of rain in one hour** against a
  sewer system designed for **1.75 inches per hour**.
- **11 people died in NYC basement apartments**; 13–14 Ida-related deaths citywide, most by
  drowning in unregulated basement units. Reporting found the basement deaths were
  concentrated among Asian residents
  (https://www.nbcnews.com/news/asian-america/ida-s-forgotten-victims-nearly-all-storm-s-basement-deaths-n1281670).
- NWS issued its first-ever flash flood emergency for New York City.

### Sandy and the VA

> Griffin et al., *A Crisis Within a Crisis: The Extended Closure of an Opioid Treatment
> Program After Hurricane Sandy*, J. Drug Issues 2018 — https://doi.org/10.1177/0022042618779541

- The Manhattan VA's opioid treatment program **closed for five months**.
- Emergency take-home doses were pre-dispensed, and guest-dosing was arranged for about
  **100 veterans** across VA and non-VA programs in NYC.

> Lukowsky et al., *Access to Care for VA Dialysis Patients During Superstorm Sandy*,
> J. Prim. Care Community Health 2019 —
> https://pmc.ncbi.nlm.nih.gov/articles/PMC6661787/ · https://doi.org/10.1177/2150132719863599

- The Manhattan VAMC **evacuated on 28 October 2012**; three hospitalised ESRD patients went
  to Brooklyn, one to the Bronx.
- The **Brooklyn campus absorbed the largest increase in dialysis encounters**; VA patients'
  dialysis encounters at non-VA facilities also rose.

> The American Legion, *VA updates Legion on Manhattan facility*, November 2012 —
> https://www.legion.org/information-center/news/veterans-healthcare/2012/november/va-updates-legion-on-manhattan-facility

### Computed in this repo, from public data

These are not quotes. They are joins you can re-run; the script is `scripts/fetch_sources.py`.

| Claim | Value | Produced by |
| --- | --- | --- |
| The Manhattan VA is in the first evacuation zone | Station **630**, Margaret Cochran Corbin VA Campus, **evacuation zone 1** | `va_facilities_nyc_hazard.parquet` — VHA facility registry × NYC hurricane evacuation zones |
| Veterans in NYC | **134,711**, of whom **53.3% are 65+** | `acs_veterans_by_zcta.parquet` — ACS 2023 5-year B21001, summed over NYC ZCTAs |
| Electricity-dependent Medicare beneficiaries in NYC | **36,146**; **3,165** on oxygen; **1,948** facility ESRD dialysis | `empower_ny_zip.parquet` — HHS emPOWER, five-borough ZIPs |
| June 2023 smoke peak | **203.5 µg/m³ PM2.5, AQI 254**, Queens monitor, 7 June 2023 | `airnow_pm25_nyc_smoke2023.parquet` — EPA AirNow daily files |
| Vulnerability stacks along the heat gradient | Utility-shutoff threat rises **5.3% → 17.2%** from HVI band 1 to 5; mobility difficulty **9.3% → 18.8%**; lacks transport **6.0% → 16.6%** | `places_zcta_nyc.parquet` × `hvi_by_zcta.parquet` |
| Most flood-exposed ZIPs | Rockaways 11692 (**59%** of area), 11693, 11694, 11697; Coney Island 11224; Lower Manhattan 10004 | `stormwater_by_modzcta.parquet` |

### Policy context

- VA, *The PACT Act and your VA benefits* — https://www.va.gov/resources/the-pact-act-and-your-va-benefits/
- VA, *Disaster help* (VSAFE fraud line, **833-388-7233**) — https://www.va.gov/resources/disaster-help/
- VA News, *Standing strong after the storm: natural-disaster fraud prevention* —
  https://news.va.gov/148481/standing-strong-after-the-storm-natural-disasters-fraud-prevention/
- Veterans Crisis Line: **dial 988, then press 1** — https://www.veteranscrisisline.net/
- VA Lighthouse Clinical Health API (FHIR R4) — https://developer.va.gov/explore/api/clinical-health
- Oracle Health Millennium FHIR R4 APIs — https://docs.oracle.com/en/industries/health/millennium-platform-apis/index.html

---

## 2. Data sources, as fetched

All of these are keyless. `scripts/fetch_sources.py` records url, rows, bytes and a sha256
prefix for each in `data/reference/manifest.json`.

| Fetcher | Endpoint | Output |
| --- | --- | --- |
| `modzcta` | https://data.cityofnewyork.us/d/pri4-ifjk | `nyc_modzcta.parquet`, `nyc_modzcta.geojson` |
| `hvi` | https://data.cityofnewyork.us/d/4mhf-duep | `hvi_by_zcta.parquet` |
| `evac` | https://data.cityofnewyork.us/d/epne-qv9x | `evac_zone_by_modzcta.parquet` |
| `stormwater` | https://data.cityofnewyork.us/Environment/NYC-Stormwater-Flood-Maps/9i7c-xyvv | `stormwater_by_modzcta.parquet` |
| `floodnet_sensors` | https://data.cityofnewyork.us/d/kb2e-tjy3 | `floodnet_sensors.parquet` |
| `floodnet_events` | https://data.cityofnewyork.us/d/aq7i-eu5q | `floodnet_events.parquet` |
| `cooling_sites` | https://data.cityofnewyork.us/d/h2bn-gu9k | `nyc_cooling_sites.parquet` |
| `empower` | https://services2.arcgis.com/ZQ4jTQn6k7VPXEwO/arcgis/rest/services/HHS_emPOWER_REST_Service_Public/FeatureServer | `empower_ny_zip.parquet` |
| `places` | https://data.cdc.gov/d/kee5-23sr | `places_zcta_nyc.parquet` |
| `svi` | https://svi.cdc.gov/Documents/Data/2022/csv/states/NewYork.csv | `svi_nyc_tract.parquet` |
| `nri` | https://services.arcgis.com/XG15cJAlne2vxtgt/arcgis/rest/services/National_Risk_Index_Census_Tracts/FeatureServer | `fema_nri_nyc_tract.parquet` |
| `va_facilities` | https://services2.arcgis.com/VFLAJVozK0rtzQmT/arcgis/rest/services/Veterans_Health_Administration_Medical_Facilities/FeatureServer | `va_facilities_ny.parquet` |
| `va_facility_hazard` | derived join | `va_facilities_nyc_hazard.parquet` |
| `airnow_smoke` | https://files.airnowtech.org/airnow/ | `airnow_pm25_nyc_smoke2023.parquet` |
| `nws_snapshot` | https://api.weather.gov/ | `nws_forecast_nyc.parquet`, `nws_alerts_ny.json` |
| `acs_veterans` | https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/ | `acs_veterans_by_zcta.parquet` |
| `synthea_sample` | https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_fhir_latest.zip | `data/raw/synthea_sample_fhir.zip` |

### Synthetic cohort

- VA, *Synthetic Suicide Prevention Dataset with SDoH* —
  https://catalog.data.gov/dataset/synthetic-suicide-prevention-dataset-with-sdoh
  Resource: https://www.data.va.gov/download/h5zp-pekf/application/zip
  **4.0 GB**, filename `csv_national_100k.zip`, root directory `csv_usa_100k/`, Synthea **CSV**
  format. CC0. The catalogue description says 10,000 records; the resource is the national
  100k CSV release. See `data/README.md`.
- Synthea sample data (FHIR R4) — https://synthetichealth.github.io/synthea-sample-data/downloads/latest/synthea_sample_data_fhir_latest.zip
- Synthea CSV data dictionary — https://github.com/synthetichealth/synthea/wiki/CSV-File-Data-Dictionary

---

## 3. Endpoints that changed — do not re-derive these

| What the spec assumed | What is actually true, 19 Sep 2026 | Use instead |
| --- | --- | --- |
| Census API is keyless for small requests | `api.census.gov` returns **302 → `/data/missing_key.html`** for every request, keyed or not | ACS Summary File, table-based, on `www2.census.gov`. Keyless. Table **B21001**. |
| AirNow API is usable with a free key | `airnowapi.org` returns `{"WebServiceError":[{"Message":"Invalid API key"}]}`; key issuance is not instant | `https://files.airnowtech.org/airnow/YYYY/YYYYMMDD/daily_data_v2.dat`, pipe-delimited, keyless, with lat/lon per monitor |
| VA Facilities API (`api.va.gov`) is open | `401 No API key found in request`; developer.va.gov needs an approved application | VHA Medical Facilities ArcGIS FeatureServer, keyless, 84 NY sites with `LAT`/`LON` |
| FEMA NRI ships a static zip at `hazards.fema.gov/nri/Content/StaticDocuments/...` | **301** to a FEMA landing page | `FEMA_NationalRiskIndex` ArcGIS FeatureServer. Note the heat-wave fields are `HWAV_*`, **not** `HRWV_*`, and `RFLD_*` is absent from the tract layer. |
| FloodNet needs a data-request form | True for `floodnet.nyc` itself | The same sensor metadata and flood events are open on NYC Open Data: `kb2e-tjy3` and `aq7i-eu5q`. No form. |
| HVI needs an NTA → ZIP crosswalk | NYC now publishes HVI **per ZCTA20** directly | `4mhf-duep`, two columns: `zcta20`, `hvi` |
| emPOWER query field is `STATE_NAME` | It is **`STATE`** (2 characters). Layer **1** is the ZIP-level all-DME layer. | `where=STATE='NY'`, 1,702 rows |
| NYC evacuation zones are 1–6 | The layer carries zones **1–7 plus a polygon coded `X`** for everywhere outside any zone. `X` is the complement, not a zone — drop it. | |
| VA Synthea release is 10,000 FHIR R4 bundles | It is a **4.0 GB Synthea CSV** release of roughly 100,000 records | Synthea's own FHIR R4 sample for the FHIR code path |

---

## 4. Event and rubric

- Health in Climate AI Hackathon NYC, 19–20 September 2026, Cornell Tech —
  https://www.healthinclimate.ai/hackathons/nyc/2026
- Tracks: *Resilient People* (health and wellbeing, including brain and mental health) and
  *Resilient Places* (built environment, infrastructure, neighbourhoods).
- Prior-year rubric — https://health-in-climate-ai-hackathon.devpost.com/

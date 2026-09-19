# cohort


## 17:00 — cohort/build.py
10,000 veterans in 175 ZIPs; ZIP·sex·age from ACS B21001 (r=0.97), 10 PLACES rates within 5% of source, emPOWER equipment, 107 OTP (81 at station 630). Fixed ACS fetcher double-counting 46 Miami-Dade tracts (NYC vets 134,711 → 131,195). Meds empty until medications.py; race/ethnicity null; only 7 on dialysis (emPOWER rate). make check green.

## 19:12 — cohort/medications.py
Medication columns are live: 10k cohort now 4.4 active meds each, 72% heat-impairing, 18% on the CDC RAAS+diuretic pair, 11% cold-chain, 5.5% controlled — `pharmacist_med_review` (1,111) and `controlled_substance_bridge` (3,051) fire for the first time. Every list is a real Synthea patient's, bootstrapped whole from new `data/reference/synthea_med_profiles.parquet` (109 rows, 5 KB, committed) within age band, so a clean clone needs no `data/raw/`. **Most specific VA class only** — HCTZ is CV701 not also CV700 — or every parented drug's weight and ACB doubles; hierarchy read off the crosswalk, not hand-written. Corrected three docs against the data: README/SPEC/BUILD_PLAN said 77% heat (that is the any-hazard share; heat is 65%) and 5% ACB≥3 (it is 12%). 29 tests, make check green.

Two for other lanes: `cold_chain_plan` is eligible for 1,083 veterans but the greedy allocator still never selects it (api lane — `care_team_call` appears to dominate the same capacity unit). The medication draw ignores the veteran's own diagnosis list, so cold-chain does not imply `diabetes`; six diabetics in the sample is too few to condition on, and it closes with the Synthea swap.

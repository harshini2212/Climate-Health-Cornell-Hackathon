# Roadmap

*As of October 2026. The order is fixed; the dates are targets.*

Leeward works on synthetic veterans in NYC. The path from here is: a clean codebase, a model
calibrated on real public data, a second region, then a pilot that runs **inside** a partner's
environment, so that no veteran data ever leaves it.

## Where it stands

- The decision layer and table contracts are solid and heavily tested.
- The rung-1 model has been fitted (r-hat 1.008), but the demo still scores with the prior.
- All outcomes are simulated from the same design the model uses. The headline "harm averted"
  result therefore shows the machinery works, not that it helps real veterans.
- On the Brier skill score, three of the five needs (breathing, mental health, access loss)
  are no better than a constant.

## Phase 0: cleanup (October 2026)

No behaviour changes:

- one parallel test run in the gate, with timing tests moved to `make perf`;
- a locked toolchain, pre-commit, mypy, and a UI build in CI;
- import layering enforced, so production code never imports the synthetic cohort;
- demo constants moved into settings;
- documentation for a growing team.

## Phase 1: research-grade v1 (November 2026 – January 2027)

- **Region abstraction.** Replace the NYC-only `modzcta` key with `region_id` + `geo_id`,
  configured per region. First new regions: one hot inland market and one flood-prone market.
- **Real calibration targets.** NYC heat ED and hospitalisation data, NWS HeatRisk, daily PRISM
  temperature, and EAGLE-I power outages.
- **Synthetic cohort.** Regenerate it from Synthea's veteran modules (PTSD, depression,
  self-harm, substance use, cancers, dialysis, homelessness).
- **Model.** Fit rung 2, and make a fitted model drive the product. Improve the weak needs or
  drop them from v1.
- **Two tracks, chosen by the evidence:**
  - heat with behavioral health and heat-sensitive medications;
  - floods and storms with cancer and other site-dependent treatment.
- **OMOP input adapter.** Tested on Synthea exported to OMOP, so the code that would run inside
  a partner exists before the partner does.
- **Platform:**
  - Postgres for the outcome log, actions and audit trail;
  - a versioned model registry;
  - nightly jobs that run ingest, then scoring, then allocation;
  - a versioned API;
  - removing the UI's silent fallback to fixture data;
  - a data class on every column.

## Phase 2: pilot product (February – summer 2027)

- Containers that run inside the partner's environment, behind their single sign-on.
- Role-based access, and an append-only log of who viewed which veteran.
- A security package; drift, calibration and fairness monitoring.
- Outreach drafted by Leeward and sent by a person.
- Shadow mode for one summer before any list is acted on.

## Before any real data

`docs/security/phi-design.md` must exist and be approved first. ZIP codes plus exact dates
already make veteran-day rows a HIPAA limited data set, not de-identified data.

# Leeward — veteran care continuity under climate events

## What this is
A Bayesian daily-hazard model plus a capacity-aware decision layer that tells VA care teams
which veterans to reach before a climate event, when, and with what action. It began as a
hackathon project (Health in Climate AI Hackathon NYC 2026, second place) and is now being
built toward a VA pilot. Synthetic people, real places.

**No PHI and no real veteran data**, in the repo or in any code path, until
`docs/security/phi-design.md` exists and the team has approved it. That rule changes only by a
deliberate, reviewed PR, never as a side effect of another task.

Read before coding, in this order:
1. `data/README.md` — what data already exists and where every rate comes from
2. `docs/SPEC.md` — contracts, the model, the decision layer
3. `docs/ROADMAP.md` — what is being built next, and why
4. `CONTRIBUTING.md` — branches, PRs and the gate

## The gate
`make check` runs the same gate steps as CI: lint, mypy, import layering, and every test once in
parallel (`pytest -n auto`), including the semantic guardrails in
`tests/guardrails/test_guardrails.py`. **Red means the PR does not merge.** Run it before you
claim anything is done. A green gate is necessary, not sufficient: every PR is also reviewed.

The three wall-clock tests (slider < 300 ms, scoring < 5 s, toy fit < 60 s; CI runners get
looser bounds) are marked `perf` and run with `make perf`, on a quiet machine. CI also runs
them, non-blocking, plus nightly on `main`. `make setup` installs exactly `uv.lock`, like CI.
`make fixtures` regenerates correctly-shaped fake data for every contract table.

Tests live in `tests/{unit,contracts,guardrails,demo}/`. A guardrail for a module that does
not exist yet skips with a message naming what it will enforce, and starts enforcing the
moment that module lands. Before you build a module, read the skipping guardrail for it — **it
is your specification.**

## The data is already fetched
`data/reference/` holds 22 joined, verified public tables (~4 MB, committed), built by
`scripts/fetch_sources.py` and catalogued in `data/README.md`.

**Never invent a neighbourhood rate that exists in `data/reference/`.** Mobility impairment,
social support, utility-shutoff risk, transport barriers, chronic-disease prevalence and
powered-equipment counts are all real per-ZIP numbers from CDC PLACES, emPOWER, ACS and NYC
Open Data. Read them.

The same goes for medication. Synthea puts an RxNorm code on every prescription;
`va_drug_class_members.parquet` maps those to the VA's own drug classes and
`med_climate_risk.csv` attaches CDC's mechanism, weight, ACB score and the
controlled / cold-chain / narrow-TI flags. Do not hand-write a drug list.

What is legitimately synthetic — housing floor, burn-pit years, PTSD severity, per-person AC,
mail-order status, days of supply remaining, and every daily outcome — carries a
`_synthetic` flag.

## Contracts
A contract changes only in a PR that edits `docs/SPEC.md` §3 in the same change and says so in
its description.

- `data/cohort.parquet`, `hazards.parquet`, `site_status.parquet`, `outcomes.parquet`,
  `scores.parquet`, `actions.parquet`, `outcome_log.parquet`: columns in `leeward/schema.py`.
- `data/posterior.nc`: ArviZ InferenceData; var names match `leeward/model/priors.py`.
- API bodies/responses: pydantic models in `leeward/api/schemas.py`. Models another layer also
  builds (`Message`, `ReportResponse` and its rows) live in `leeward/contracts/` and are
  re-exported from `api/schemas.py` unchanged.
- `leeward/model/design.py` is shared by the simulator and the model. Do not fork it.
- Write tables with `schema.write(df, "<table>")`, never `df.write_parquet(...)`. It validates
  first.
- **`(region_id, geo_id)` is the geography key everywhere.** Not `zip`, not `zcta`, not NTA,
  and not `modzcta`, which `schema.write()` now refuses. A place is a `regions/<id>.yaml`
  read through `leeward/geo/region.py`; code never names NYC's tables or boroughs directly.
- **Layering:** production code never imports `leeward.cohort` or `leeward.eval`, and only
  `leeward.api` imports `leeward.api`. `lint-imports` enforces both, in `make check` and CI.
  Deploy-time defaults (scenario, CORS, OTP stations) are in `leeward/settings.py`
  (`LEEWARD_*` env, no `.env`).

## Stack
Python 3.11, NumPyro + JAX (CPU), polars, FastAPI, React + Vite + deck.gl. Dependencies are
locked in `uv.lock`. Never run inference inside a request.

## Rules
- **No API keys.** A clean clone with no `.env` must produce a working demo. `make demo` must
  not touch the network at all; it reads `data/reference/` and cached parquets.
- Every real number shown anywhere is in `docs/sources.md` with a URL. Synthetic numbers say so.
- Build the model as a ladder (SPEC §6.0). Say which rung actually fitted, and its r-hat.
- Drivers come from posterior contributions; no SHAP.
- Fairness audit runs in `make report`; a failing audit is displayed, never suppressed.
- **Leeward never changes a medication.** Medication actions flag a veteran for the VA
  clinical pharmacist, who decides. `pharmacist_slot` is a scarce capacity unit because it is
  a real person's afternoon.
- Every outreach message includes: VA channel tag, 4-word verification phrase,
  "The VA will never ask you to pay, wire money, or share bank details",
  VSAFE 833-388-7233, and "Veterans Crisis Line: dial 988, press 1".
- Outreach is drafted by Leeward and sent by a person. No code path sends a message to a
  veteran on its own.
- Seed everything. The same click must produce the same number every time.
- For a bug, write the failing test first, then fix it.
- `make demo` boots in < 60 s from a clean clone.

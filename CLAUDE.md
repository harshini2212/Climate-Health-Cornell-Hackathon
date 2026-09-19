# Leeward — veteran care continuity under climate events

## What this is
Bayesian daily-hazard model + capacity-aware decision layer for VA care teams, for the
Health in Climate AI Hackathon NYC 2026. Synthetic people, real places. Never write code
that could ingest real PHI.

Read before coding, in this order:
1. `data/README.md` — what data already exists and where every rate comes from
2. `docs/SPEC.md` — the section for your lane
3. `docs/BUILD_PLAN.md` — who owns what, and what is being cut

## The data is already fetched
`data/reference/` holds 17 joined, verified public tables (~4 MB, committed), built by
`scripts/fetch_sources.py` and catalogued in `data/README.md`.

**Never invent a neighbourhood rate that exists in `data/reference/`.** Mobility impairment,
social support, utility-shutoff risk, transport barriers, chronic-disease prevalence and
powered-equipment counts are all real per-ZIP numbers from CDC PLACES, emPOWER, ACS and NYC
Open Data. Read them. What is legitimately synthetic — housing floor, burn-pit years, PTSD
severity, per-person AC, every daily outcome — carries a `_synthetic` flag.

## Contracts (frozen; change only by editing docs/SPEC.md §3 and telling the other person)
- `data/cohort.parquet`, `hazards.parquet`, `site_status.parquet`, `outcomes.parquet`,
  `scores.parquet`, `actions.parquet`, `outcome_log.parquet`: columns in `leeward/schema.py`.
- `data/posterior.nc`: ArviZ InferenceData; var names match `leeward/model/priors.py`.
- API bodies/responses: pydantic models in `leeward/api/schemas.py`.
- `leeward/model/design.py` is shared by the simulator and the model. Do not fork it.
- **`modzcta` is the geography key everywhere.** Not `zip`, not `zcta`, not NTA.

Contract-file ownership: `leeward/schema.py` and `model/design.py` belong to the `cohort`
lane; `api/schemas.py` belongs to the `api` lane. Do not edit a contract file you do not own —
ask the owner and they will push within five minutes.

## Stack
Python 3.11, NumPyro + JAX (CPU), polars, FastAPI, React + Vite + deck.gl.
Makefile: `fixtures | data | cohort | fit | score | demo | report | test`.
Never run inference inside a request.

## Rules
- **No API keys.** A clean clone with no `.env` must produce a working demo. `make demo` must
  not touch the network at all; it reads `data/reference/` and cached parquets.
- Every real number shown anywhere is in `docs/sources.md` with a URL. Synthetic numbers say so.
- Build the model as a ladder (SPEC §6.0). Rung 0 is prior-only and is built first. Say which
  rung actually fitted, and its r-hat.
- Drivers come from posterior contributions; no SHAP.
- Fairness audit runs in `make report`; a failing audit is displayed, never suppressed.
- Every outreach message includes: VA channel tag, 4-word verification phrase,
  "The VA will never ask you to pay, wire money, or share bank details",
  VSAFE 833-388-7233, and "Veterans Crisis Line: dial 988, press 1".
- `pytest -q` passes before any merge; `make demo` boots in < 60 s from a clean clone.
- One task per prompt; write the acceptance test first; ask before touching a contract file.
- Seed everything. The same click must produce the same number in rehearsal and on stage.

## Lanes
Two builders, six git worktrees, one `main`.

| Owner | Lanes |
| --- | --- |
| Rahul — runs the live demo | `api`, `ui`, `demo` |
| Partner — owns the numbers | `cohort`, `model`, `eval` |

Branch per lane (`lane/api`, `lane/ui`, …). Merge to `main` through a green `pytest -q` every
60–90 minutes, never by hand-copying files.

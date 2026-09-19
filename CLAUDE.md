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

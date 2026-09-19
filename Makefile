.PHONY: help setup sources sources-heavy fixtures hazards cohort fit score demo report \
        test check status smoke clean-clone lanes
PY  ?= .venv/bin/python
PIP ?= .venv/bin/python -m pip

help:               ## show this
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

# --------------------------------------------------------------------------- #
# The gate. If `make check` is green you may merge. If it is red you may not.
# Nobody reads diffs on this project; this is the review.
# --------------------------------------------------------------------------- #

check:              ## THE GATE: lint + every test + the guardrail to-do list
	@echo "── lint ──────────────────────────────────────────────"
	@.venv/bin/ruff check leeward scripts tests || (echo "ruff failed"; exit 1)
	@echo "── tests ─────────────────────────────────────────────"
	@.venv/bin/pytest -q -p no:cacheprovider
	@echo
	@echo "── guardrails still waiting on unbuilt modules ───────"
	@.venv/bin/pytest -q -rs -p no:cacheprovider 2>/dev/null \
	  | grep -E '^SKIPPED' | sed 's/SKIPPED \[[0-9]*\] /  · /' || echo "  (none — everything is built)"
	@echo
	@echo "GREEN. Safe to merge."

status:             ## where the build actually is, without reading any code
	@bash scripts/status.sh

test:               ## pytest only
	@.venv/bin/pytest -q

# --------------------------------------------------------------------------- #
# Setup and data
# --------------------------------------------------------------------------- #

setup:              ## venv + deps + editable install
	uv venv --python 3.11 .venv || python3.11 -m venv .venv
	uv pip install -e ".[dev,geo]" || $(PIP) install -e ".[dev,geo]"
	@echo "Now run: make fixtures && make check"

sources:            ## re-fetch public data into data/reference/ (needs network; already committed)
	$(PY) scripts/fetch_sources.py

sources-heavy:      ## + stormwater GIS, ACS summary file, Synthea sample
	$(PY) scripts/fetch_sources.py --heavy

fixtures:           ## correctly-shaped fake data for every contract table
	$(PY) scripts/make_fixtures.py

# --------------------------------------------------------------------------- #
# The pipeline. Each target replaces one fixture with the real thing.
# --------------------------------------------------------------------------- #

hazards:            ## scenario YAML + data/reference -> hazards.parquet, site_status.parquet
	$(PY) -m leeward.ingest.hazards --scenario scenarios/sandy_then_heat.yaml

cohort: hazards     ## 10k synthetic veterans + truth.json + 120 days of outcomes
	$(PY) -m leeward.cohort.build
	$(PY) -m leeward.cohort.simulate

fit:                ## NumPyro NUTS -> data/posterior.nc  (see SPEC 6.0 for the rung)
	$(PY) -m leeward.model.fit

score:              ## posterior x hazards -> scores.parquet, actions.parquet
	$(PY) -m leeward.model.score
	$(PY) -m leeward.decision.allocate

report:             ## full eval harness incl. the fairness audit -> report/report.json
	$(PY) -m leeward.eval.report

# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

demo:               ## boot API + UI. Must work with the network off.
	$(PY) -m uvicorn leeward.api.main:app --port 8000 & \
	  cd ui && npm run dev

smoke:              ## boot the API and hit every route; fails if any shape is wrong
	@bash scripts/smoke_demo.sh

clean-clone:        ## prove a fresh clone boots offline in under 60s
	@bash scripts/clean_clone_test.sh

lanes:              ## create the six git worktrees
	@for l in api ui demo cohort model eval; do \
	  git worktree add ../lw-$$l -b lane/$$l 2>/dev/null && echo "  ../lw-$$l" || echo "  ../lw-$$l exists"; \
	done
	@echo "Each lane: cd ../lw-<lane> && claude"

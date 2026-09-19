.PHONY: help setup sources fixtures hazards cohort fit score demo report test clean-clone
PY ?= .venv/bin/python
PIP ?= .venv/bin/python -m pip

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | expand -t22

setup:              ## create venv and install
	uv venv --python 3.11 .venv || python3.11 -m venv .venv
	uv pip install -e ".[dev]" || $(PIP) install -e ".[dev]"

sources:            ## re-fetch public data into data/reference/ (needs network)
	$(PY) scripts/fetch_sources.py

sources-heavy:      ## + stormwater GIS, ACS summary file, Synthea sample
	$(PY) scripts/fetch_sources.py --heavy

fixtures:           ## fake-but-correctly-shaped parquets so every lane can start
	$(PY) scripts/make_fixtures.py

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

demo:               ## boot API + UI. Must work with the network off.
	$(PY) -m uvicorn leeward.api.main:app --port 8000 & \
	  cd ui && npm run dev

report:             ## full eval harness incl. the fairness audit -> report/report.json
	$(PY) -m leeward.eval.report

test:               ## pytest
	.venv/bin/pytest -q

clean-clone:        ## prove a fresh clone boots offline in under 60s
	bash scripts/clean_clone_test.sh

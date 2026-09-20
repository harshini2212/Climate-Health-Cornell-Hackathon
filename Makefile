.PHONY: help setup sources sources-heavy fixtures hazards cohort fit score demo report ablate \
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

prompt:             ## print a prompt to paste:  make prompt N=4   (no N lists them)
	@bash scripts/prompt.sh $(N)

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

# Scoring is two stages, the same two `scripts/baseline.py` records. `model/score.py` -- the
# posterior scorer -- takes over from the rung-0 `model/score_prior.py` the day it lands, with
# no edit here. baseline.py names its stages outright, so that one does need an edit; whoever
# lands score.py should make the two agree, or a recorded row describes a run nobody made.
SCORER := $(if $(wildcard leeward/model/score.py),leeward.model.score,leeward.model.score_prior)

# `make fit` has nothing to run until the model lane lands `model/fit.py` (SPEC 6.0, rung 1).
# It still fails -- a target that claims success without fitting anything is worse -- but it
# fails saying which rung is built, instead of `No module named` at 2am.
fit:                ## NumPyro NUTS -> data/posterior.nc  (see SPEC 6.0 for the rung)
ifeq ($(wildcard leeward/model/fit.py),)
	@echo "make fit: nothing to fit. The model is at rung 0 (prior-only), so leeward/model/fit.py has to land before there is a posterior to cache; until then \`make score\` is the whole model." >&2
	@exit 1
else
	$(PY) -m leeward.model.fit
endif

score:              ## hazards x the model that exists -> scores.parquet, actions.parquet
	$(PY) -m $(SCORER)
	$(PY) -m leeward.decision.allocate

baseline:           ## run the pipeline and record one row in docs/BASELINES.md
	$(PY) scripts/baseline.py $(if $(LABEL),--label "$(LABEL)",)

baseline-diff:      ## diff the two most recent baselines, record nothing
	$(PY) scripts/baseline.py --compare

report:             ## full eval harness incl. the fairness audit -> report/report.json
	$(PY) -m leeward.eval.report

# Separate from `report` on purpose: six models over the real cohort is about 30 seconds,
# and `make report` is run every few edits. It caches report/ablations.json, which the next
# `make report` picks up. Re-run it whenever the model or the decision layer moves.
ablate:             ## what each block of the model is worth -> report/ablations.{csv,json,html}
	$(PY) -m leeward.eval.ablate

# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #

# Which scenario day the board opens on. Unset -- the normal case -- means the UI asks for
# no day and takes the window `leeward/demo.py` picks: two days in front of landfall. Set it
# to replay a different beat by hand: `make demo DAY=0` is the calm week nine weeks earlier.
DAY ?=

demo:               ## boot API + UI, offline. `make demo DAY=0` opens on the calm week.
	$(PY) -m uvicorn leeward.api.main:app --port 8000 & \
	  cd ui && VITE_DEMO_DAY="$(DAY)" npm run dev

smoke:              ## boot the API and hit every route; fails if any shape is wrong
	@bash scripts/smoke_demo.sh

clean-clone:        ## prove a fresh clone boots offline in under 60s
	@bash scripts/clean_clone_test.sh

lanes:              ## create the six git worktrees, each with its own venv
	@for l in api ui demo cohort model eval; do \
	  if [ -d ../lw-$$l ]; then echo "  ../lw-$$l exists"; else \
	    git worktree add -q ../lw-$$l -b lane/$$l && echo "  ../lw-$$l created"; fi; \
	  if [ ! -x ../lw-$$l/.venv/bin/python ]; then \
	    echo "    installing venv..."; \
	    (cd ../lw-$$l && uv venv --python 3.11 .venv -q && uv pip install -q -e ".[dev]" \
	      && .venv/bin/python scripts/make_fixtures.py >/dev/null) ; fi; \
	done
	@echo
	@echo "Each lane needs its OWN venv -- an editable install resolves to wherever it"
	@echo "was installed from, so a shared venv would test the wrong checkout silently."
	@echo
	@echo "Start a lane:  cd ../lw-<lane> && claude"
	@echo "Then paste that lane's Wave 1 prompt from docs/PROMPTS.md"

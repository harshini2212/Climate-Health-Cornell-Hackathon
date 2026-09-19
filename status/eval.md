# eval


## 16:57 — eval/decision_quality.py
Harm averted at K∈{20,40,80} calls/day, Leeward vs age / n_chronic / random(seed 0), same K-call budget for all → report/decision_quality.{csv,html}; baselines + scoring tested, Leeward-beats-random skips only on the missing `leeward.decision.allocate` and needs `decision.severity.SEVERITY` + `decision.tau.TAU` from the api lane. make check green.

## 19:0x — tests no longer read data/
`tests/tables.py` builds every contract frame in memory from the `scripts/make_fixtures.py`
generators (seeded, geography still real from `data/reference/`); the five copies of the
`data/*.parquet` reader are gone, and a new guardrail fails any test file that adds a sixth.
Suite is now identical with fixtures, with the real 10k cohort, and with `data/` empty.
Fixed two assertions that were wrong for real data: a day is in the action list exactly when
someone is above the Everyday floor (the real Sandy run has 14 calm days that correctly
produce nothing, now pinned by two explicit tests), and the borough floor derives
floor(0.2 x 20) = 4 instead of hard-coding it.

# model

## 17:05 — model rung 0 (priors.py, design.py, score_prior.py)
Prior-only, no MCMC: 400 draws over 35 terms (SPEC 6.3 interactions + SiteDown) -> scores.parquet, model_rung 0; 10k x 7 days in 0.6s, 10k x 120 in 9s. Run `python -m leeward.model.score_prior` (`make score` does not call it yet). make check green.

## 19:05 — the simulator (cohort/simulate.py, data/truth.json)
10k veterans x 120 days x 5 needs in 1.2s through `design.py` (not forked): off-event rates 0.30–1.10%/day, heat peaks 16.1% on 8 Aug, dialysis patients of closed station 630 run 56% treatment-gap days. Outcomes are seeded per calendar day, so a date draws the same outcome in any window.
`truth.json` carries all 35 design terms with a source line each — SPEC 5.4 where it fixed a number, an assumption line where it did not — including the six medication interactions, which stay dormant until `medications.py` fills those columns and then switch on by themselves.

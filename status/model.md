# model

## 17:05 — model rung 0 (priors.py, design.py, score_prior.py)
Prior-only, no MCMC: 400 draws over 35 terms (SPEC 6.3 interactions + SiteDown) -> scores.parquet, model_rung 0; 10k x 7 days in 0.6s, 10k x 120 in 9s. Run `python -m leeward.model.score_prior` (`make score` does not call it yet). make check green.

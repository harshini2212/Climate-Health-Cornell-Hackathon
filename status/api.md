# api


## 17:10 — decision/{severity,tau,eha,tiers,allocate}.py
allocate() + total_eha(): greedy by EHA per bucket, act-now 3 slots valued on residual risk (never >100% of a need), capacity-monotone by construction (72k-case search, 0 violations; the lazy re-valuing variant failed and its seeds are pinned). 10k veterans/day in ~105 ms. tiers.py is score-only: hazard-triggered act-now rules still TODO. make check green.

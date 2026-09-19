# api


## 17:10 — decision/{severity,tau,eha,tiers,allocate}.py
allocate() + total_eha(): greedy by EHA per bucket, act-now 3 slots valued on residual risk (never >100% of a need), capacity-monotone by construction (72k-case search, 0 violations; the lazy re-valuing variant failed and its seeds are pinned). 10k veterans/day in ~105 ms. tiers.py is score-only: hazard-triggered act-now rules still TODO. make check green.

## 19:20 — api/main.py, api/store.py, outreach/export.py, allocate.compare()
All eight routes over cached parquets, POST /actions 127 ms worst case at 10,000 veterans x 120 days (budget 300); /message is 503 until outreach.messages lands; scenario and prior_scale other than sandy_then_heat / 1.0 are refused, not faked. smoke_demo.sh passes offline, make check green.

# api


## 17:10 — decision/{severity,tau,eha,tiers,allocate}.py
allocate() + total_eha(): greedy by EHA per bucket, act-now 3 slots valued on residual risk (never >100% of a need), capacity-monotone by construction (72k-case search, 0 violations; the lazy re-valuing variant failed and its seeds are pinned). 10k veterans/day in ~105 ms. tiers.py is score-only: hazard-triggered act-now rules still TODO. make check green.

## 19:20 — api/main.py, api/store.py, outreach/export.py, allocate.compare()
All eight routes over cached parquets, POST /actions 127 ms worst case at 10,000 veterans x 120 days (budget 300); /message is 503 until outreach.messages lands; scenario and prior_scale other than sandy_then_heat / 1.0 are refused, not faked. smoke_demo.sh passes offline, make check green.

## 20:00 — allocate.headline, api/schemas.ActionRow, Week.tsx, ui fixtures
The queue card's reason line is now `headline` ("A gap in treatment, 25% chance — act today"): need, honest probability, tier urgency, and no condition, medicine or service. `rationale` and `top_driver` are untouched and still name all three for CareTeam.tsx and VeteranCard.tsx, so the drill-down lost nothing — test_the_drilldown_keeps_the_rich_rationale holds that line. xfail(strict) marker removed; the board test now also bans the string `rationale` from Week.tsx (comments stripped first) so nobody puts it back.
Partner/cohort: `headline` rides along as an allow_extra column on `actions`, **not** a contract column — CLAUDE.md gives schema.py to the cohort lane and Table.allow_extra documents exactly this, so no contract edit and nothing for you to push. It is a required field on `api.ActionRow`, so /actions fails loudly if the allocator ever stops emitting it. Also note: the ui fixtures were stale (built for 2026-06-03, a data build that no longer exists); regenerated with `--date 2026-07-16`, which is test_api's `DAY`. make_ui_fixtures defaults to `actions["date"][0]` = 2026-07-01 now, so the demo day is unpinned — not my task, but somebody should pin it.

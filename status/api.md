# api


## 17:10 — decision/{severity,tau,eha,tiers,allocate}.py
allocate() + total_eha(): greedy by EHA per bucket, act-now 3 slots valued on residual risk (never >100% of a need), capacity-monotone by construction (72k-case search, 0 violations; the lazy re-valuing variant failed and its seeds are pinned). 10k veterans/day in ~105 ms. tiers.py is score-only: hazard-triggered act-now rules still TODO. make check green.

## 19:20 — api/main.py, api/store.py, outreach/export.py, allocate.compare()
All eight routes over cached parquets, POST /actions 127 ms worst case at 10,000 veterans x 120 days (budget 300); /message is 503 until outreach.messages lands; scenario and prior_scale other than sandy_then_heat / 1.0 are refused, not faked. smoke_demo.sh passes offline, make check green.

## 20:00 — allocate.headline, api/schemas.ActionRow, Week.tsx, ui fixtures
The queue card's reason line is now `headline` ("A gap in treatment, 25% chance — act today"): need, honest probability, tier urgency, and no condition, medicine or service. `rationale` and `top_driver` are untouched and still name all three for CareTeam.tsx and VeteranCard.tsx, so the drill-down lost nothing — test_the_drilldown_keeps_the_rich_rationale holds that line. xfail(strict) marker removed; the board test now also bans the string `rationale` from Week.tsx (comments stripped first) so nobody puts it back.
Partner/cohort: `headline` rides along as an allow_extra column on `actions`, **not** a contract column — CLAUDE.md gives schema.py to the cohort lane and Table.allow_extra documents exactly this, so no contract edit and nothing for you to push. It is a required field on `api.ActionRow`, so /actions fails loudly if the allocator ever stops emitting it. Also note: the ui fixtures were stale (built for 2026-06-03, a data build that no longer exists); regenerated with `--date 2026-07-16`, which is test_api's `DAY`. make_ui_fixtures defaults to `actions["date"][0]` = 2026-07-01 now, so the demo day is unpinned — not my task, but somebody should pin it.

## 00:50 — tau.yaml lead_days, schema.actions.lead_days, allocate by do-by day
An action now has a day it must be DONE by. `tau.yaml` gained a `lead_days:` section beside
`tau:` (both sections required; a file with only one is refused, so dropping lead times cannot
quietly mean "everything is same-day"). 2 days for the pre-arranged actions — `alt_site_booking`,
`early_refill`, `switch_to_local_pickup`, `backup_power_plan`, `cold_chain_plan`,
`pharmacist_med_review`, `controlled_substance_bridge`; 1 for rides and evacuation transport;
0 for the call, the text and the check-in. `actions.date` is now the **do-by** day and
`lead_days` is a contract column (schema.py, SPEC §3.5 + §7.4 updated), so a Wednesday surge
spends Monday's booking slot. `risk_date = date + lead_days` rides along as an allow_extra
column and is a required field on `api.ActionRow`, next to `lead_days` — **ui: the week ribbon
can draw both marks now.** `ActionsResponse.n_too_late` is new: actions whose do-by day was
already behind the day being planned, allocated in their own counterfactual day so they never
eat the real budget. `compare()` returns a third value for it; `allocate()` is unchanged.
`allocate.LEAD` (rationale phrasing) renamed `INSTRUCTION` so it stops colliding with the days.

Two slot budgets, not one, and this is the part that nearly shipped wrong: spreading a risk
day's actions over three do-by days let a veteran collect up to 7 of them and report **10% more
harm averted than they carry** — `_values` prices marginally per risk day, so that budget has
to survive. Pools are therefore filled as one queue with per-(veteran, risk day) *and*
per-(veteran, do-by day) limits. Total EHA on the panel: 6,434 scheduled against 7,102 same-day,
a 9.4% cost, inside the 15% bar; the test pins it. `test_guardrails.one_action_per_veteran`
now says which day it means — it was false for real output grouped by do-by day (101 cases).
Cost at 10k veterans: 1.31x the old route, measured interleaved A/B (the machine is at load
100+ from other lanes, so absolute ms are meaningless right now; the 300 ms test is the gate).
Also: `ui/public/fixtures/*` rebuilt — they carry `lead_days`/`risk_date` **and** a newer
`scores.parquet` than they were built from, so every offline number moved, not just the new
columns. `report.json` deliberately left at the committed one; regenerating it here would have
wiped `decision_quality` (this worktree has no `report/report.json`).

One number to know before the stage: `n_too_late` on the demo day (2026-08-01, slider at 100
calls) is **747**, and it is capacity-shaped, not lateness-shaped — it is what two days of the
team's booking capacity would have held. Today's risk always needs work that was due two days
ago, so it is non-zero on every day that carries risk. Say it as "two days of this team's
capacity could never be offered to us", not as "747 people were failed".

**Unverified on merge:** `test_the_slider_answers_inside_300ms_at_ten_thousand_veterans` was
the one red test and could not be evaluated — another session was holding the machine at
627% CPU, where unmodified `main` also fails it (488 ms median, budget 300). Everything else
is green: 857 passed, 2 skipped. **Re-run `make check` on a quiet machine before the demo.**

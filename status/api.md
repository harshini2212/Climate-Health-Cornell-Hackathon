# api


## 17:10 — decision/{severity,tau,eha,tiers,allocate}.py
allocate() + total_eha(): greedy by EHA per bucket, act-now 3 slots valued on residual risk (never >100% of a need), capacity-monotone by construction (72k-case search, 0 violations; the lazy re-valuing variant failed and its seeds are pinned). 10k veterans/day in ~105 ms. tiers.py is score-only: hazard-triggered act-now rules still TODO. make check green.

## 19:20 — api/main.py, api/store.py, outreach/export.py, allocate.compare()
All eight routes over cached parquets, POST /actions 127 ms worst case at 10,000 veterans x 120 days (budget 300); /message is 503 until outreach.messages lands; scenario and prior_scale other than sandy_then_heat / 1.0 are refused, not faked. smoke_demo.sh passes offline, make check green.

## 20:00 — allocate.headline, api/schemas.ActionRow, Week.tsx, ui fixtures
The queue card's reason line is now `headline` ("A gap in treatment, 25% chance — act today"): need, honest probability, tier urgency, and no condition, medicine or service. `rationale` and `top_driver` are untouched and still name all three for CareTeam.tsx and VeteranCard.tsx, so the drill-down lost nothing — test_the_drilldown_keeps_the_rich_rationale holds that line. xfail(strict) marker removed; the board test now also bans the string `rationale` from Week.tsx (comments stripped first) so nobody puts it back.
Partner/cohort: `headline` rides along as an allow_extra column on `actions`, **not** a contract column — CLAUDE.md gives schema.py to the cohort lane and Table.allow_extra documents exactly this, so no contract edit and nothing for you to push. It is a required field on `api.ActionRow`, so /actions fails loudly if the allocator ever stops emitting it. Also note: the ui fixtures were stale (built for 2026-06-03, a data build that no longer exists); regenerated with `--date 2026-07-16`, which is test_api's `DAY`. make_ui_fixtures defaults to `actions["date"][0]` = 2026-07-01 now, so the demo day is unpinned — not my task, but somebody should pin it.

## 01:20 — lead time: actions are scheduled by the day they must be DONE
`tau.yaml` now carries `lead_days` beside every τ row (docs/proposal.md §6's 5/2/0-day catalog): `alt_site_booking` 2, `early_refill`/`cold_chain_plan`/`pharmacist_med_review` 5, `switch_to_local_pickup`/`backup_power_plan`/`controlled_substance_bridge` 2, the two rides 1, `care_team_call`/`verified_text`/`check_in_call`/`evacuation_assist` 0 (a flash flood gives no notice, so evacuation help must still be offerable on the day). `tau.load()` is unchanged; `tau.leads()` is the new reading of the same file.

**`actions.date` is now the do-by day, and `actions.lead_days` is a contract column.** The risk day is `date + lead_days`. Capacity is spent on the do-by day, so one work day's 40 calls cover its own risk, the bookings two days out and the refills five days out at once — a live `POST /actions` for 2026-07-15 returns 347 lead-0, 15 lead-1, 155 lead-2 and 107 lead-5 rows. Do-by days share nothing, so the route is still one request per day: it reads `date .. date + 5` and returns what has to be done on `date`.

**Rahul / `ui`:** `ActionRow.lead_days: int` is required and `ActionsResponse.n_too_late: int` is new. The ribbon can now put two marks on a card — the do-by day is the response's `date`, the risk day is `date + lead_days`. `n_too_late` is the count of actions for the days ahead whose do-by day is already behind `date`; it is worth showing ("9 actions were already too late to book when the forecast arrived"). `ui/src/lib/types.ts` is yours and I did not touch it; `ui/public/fixtures/actions_candidates.json` is regenerated because `test_ui_fixtures` validates every candidate against `ActionRow`.

**Partner / `cohort`, `eval`:** I took `leeward/schema.py` this round for the one column, as prompt 13 directed — `lead_days`, Int32, bounds (0, 30). Nothing else in it moved. `compare()` now returns three values, `(actions, baseline_totals, n_too_late)`; `allocate()` is unchanged. `allocate(..., date=D)` now means "the work list for day D", not "the scores for day D", which is what `decision_quality.leeward_actions` already wants: given one day of scores it offers only the same-day actions, which is the like-for-like K-call comparison it was written for.

Two things that moved and should not surprise anyone reading the numbers:
* The marginal value chain runs for **every** veteran now, not only Act-now ones. The schedule can hand a one-slot veteran an action on Monday and another on Wednesday for the same Wednesday risk, and standalone values would then claim more harm than the veteran carries. The hand-checked five change accordingly (V4's text .00534375 not .015; total 2.77174375 not 2.7814) — their chosen set, and what it truly averts, 2.803, are unchanged.
* Scheduling costs 0.7% of harm averted on the fixture panel, against the 15% budget in the prompt. `tests/test_schedule.py` measures it against the same code with every lead set to zero, which is the only honest way to say "versus today".

Slider: 283–320 ms at 10,000 veterans, budget 300, on a machine at load 20 with three other agents' test runs on it. It was 416–480 ms before this change on the same machine — `_values()` is vectorised (the per-veteran Python loop is gone), the greedy pass reads Python lists instead of indexing numpy scalars, and `store.leads()` stops tau.yaml being re-parsed per request. `make check` green: 873 passed, 2 skipped (both the uncommitted Synthea zip).
## 00:59 — Makefile `fit`/`score`, tests/test_make_targets.py
`make score` runs the scorer that exists (rung-0 `score_prior`, then `decision.allocate` — baseline.py's own STAGES, 1.0s on fixtures) and `make fit` exits 1 naming the rung that is built and the module that would have to land, instead of `No module named`. Both resolve through `$(wildcard ...)`: when C10 lands `model/fit.py` and `model/score.py` they are picked up with no edit — verified by dropping both in and re-running.
New gate test reads every documented target's command line out of `make -n` and asserts each module it names is findable and each script it runs exists; `make demo` is checked without booting (the `module:attr` given to uvicorn must import and have that attr, `npm run dev` must exist in ui/package.json); `make setup` is excluded, being the target that creates the environment. Docs: the three now-wrong `make fit` references fixed (README table, BUILD_PLAN T+6:00); PROMPTS.md's four `make cohort && make score` lines became true rather than wrong, so they are untouched. 855 passed, 2 skipped; red only on the 300 ms /actions perf test, 320–365 ms under load avg 17–38 from five concurrent lanes, identical at HEAD before this change.

## 02:15 — C9: FacilityStatus.date, ActionsResponse.n_not_reached, ActionsRequest.limit
All three contract gaps the week board was working around are closed. `/forecast` now returns
one `FacilityStatus` **per facility per day** (98 rows, not 14) straight off `site_status`,
which was always keyed that way — the window-level `.any()` collapse and the comment
apologising for it are gone, and the headline says when: "Site down: Margaret Cochran Corbin
VA Campus from Mon 03 Aug". `POST /actions` returns `n_not_reached`: the Act-now and Find-out
veterans a limitless team reaches with a person and this capacity reaches with nobody — a
free verified text is not being reached. It is the same number the board was computing from
a second uncapped request, and `test_n_not_reached_is_the_second_call_the_board_no_longer_has_to_make`
pins it to that definition by making both calls and comparing. `limit` cuts the rows and
nothing else: `n_selected`, `total_eha`, `counts_by_tier`, `n_not_reached`, `baselines` and
`n_too_late` all describe the whole allocation, and `store.remember` still sees every row so
`GET /message` answers for rows the caller never rendered. On the fixture panel: 630 selected
at default capacity with 125 not reached, 144 at call=10, 107 at call=80; `limit=40` gives
40 rows of 630 with the totals untouched.

`n_not_reached` costs one more greedy pass inside `compare(headroom=True)`, not a second
allocation. Every candidate is valued once and only the pass repeats — the same thing that
buys the three baselines — so it is **+5.0% on the request** (min 346 -> 364 ms, 25 interleaved
runs at 10,000 veterans on a box at load 52). A second `compare()` from the route would have
been +90%. That is the whole reason it is in the allocator and not in main.py.

**Rahul / `ui`:** three things, all independent.
1. `FacilityStatus` has a required `date`. `forecast.facilities` is 7x longer, so
   `Map.tsx:102/146/147` and `Forecast.tsx:52` will show each facility once per day and
   count "5 down" where they meant "1 down" — they need `.filter(f => f.date === <the day>)`.
   `Week.tsx:159` (`downSites`) needs the same or a dedupe. **This is a visible wrong number
   on the Map screen until you do it.** `ui/public/fixtures/forecast.json` is regenerated
   per-day from `ingest.hazards.assemble` (station 630 goes down 2026-08-03, landfall, and
   stays down); nothing else in that fixture moved and no other fixture was touched.
2. `ActionsResponse.n_not_reached` replaces the `UNCAPPED` week load. Delete the second
   `postActionsWeek` call in `Week.tsx:146` and `boardFor`'s `free` argument: 14 requests
   become 7. Definition is exactly `boardFor`'s `missed`, including that a `free` bucket
   action does not count as reached.
3. `ActionsRequest.limit` is optional and defaults to no limit, so nothing you have changes.
   Pass `limit: 40` on the week load and the board can say "40 of 6,303" from `n_selected`.
   `ui/src/lib/types.ts` is yours; I did not touch it.

**Partner / `cohort`, `model`, `eval`:** `compare()` now returns **four** values,
`(actions, baseline_totals, n_too_late, n_not_reached)`; the fourth is 0 unless you pass
`headroom=True`, which is not "nobody". `allocate()` is unchanged. I updated the seven
unpack sites in `tests/test_allocate.py` and `tests/test_schedule.py` — one `, _` each,
no assertion moved. No contract file changed: `schema.py` and `design.py` are untouched.

Gate: `make check` is red only on `test_the_slider_answers_inside_300ms...`, and it is the
machine, not this change — the model lane is running a 4-chain fit at 537% CPU and the box
is at load avg 52-57. A/B through the same test minutes apart: 498 ms median with the
headroom pass off, 520 ms with it on, both far over the 300 ms budget, i.e. red at HEAD too.
Everything else passes; re-run it when the fit finishes.

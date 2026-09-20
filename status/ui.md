# ui


## 17:40 — ui/ scaffold + screens/Map.tsx + screens/CareTeam.tsx + components/CapacitySlider.tsx
Vite+React18+TS+deck.gl; both screens render from public/fixtures (typed against api/schemas.py, tests/test_ui_fixtures.py); slider round-trip ~90 ms; npm run build clean; make check green. ZipHazard has no floodnet_trip field, so the FloodNet toggle is an outage overlay until the api lane adds it.

## 19:30 — design/template.css + App shell + screens/Forecast.tsx; Map and CareTeam restyled
Ported the Brexify command-center template (sidebar, topbar, stat strip, cards, insights, brief, tables) into ui/; new Forecast home from forecast.json + /actions; Inter bundled offline. npm run build clean; make check green. Also made test file reads explicit UTF-8 so the gate passes on Windows.

## 19:20 — screens/Week.tsx is the default screen + components/VeteranCard.tsx + components/Message.tsx
Week ribbon, banners, owner lanes and a per-day queue, de-identified to an initial-plus-id handle with names/conditions/drivers behind a click; day click opens CareTeam on that date (slider kept), card click opens the card and its message checklist. `npm run build` clean, `make check` green.
Two things to know: `ActionsRequest` is single-day so a week is 7 parallel `postActions` (plus 7 more at uncapped capacity — "not reached" cannot be read off one response), and **`/forecast` carries `FacilityStatus` without a date**, so a closure shows as a window-level strip, not on the day it starts. If the api lane adds a per-day site field the ribbon can place it properly. New fixtures `veterans.json` + `messages.json` (from the same `make_ui_fixtures.py` run) make both reveals work offline; offline the queue is one modelled day repeated and the board says so on screen.

## 19:55 — merged main; the week board now lives inside the template shell
Rebuilt Week.tsx on the template's tokens (.rb glyphs, .prog load bar, .stat lanes, .card banners; week-specific rules appended to design/leeward.css). VeteranCard and Message moved to screens/ and now fill the two ComingSoon slots the shell already reserved; both use the shared lib/labels.ts instead of their own label maps. CareTeam keeps the template rewrite plus an optional `date` prop so a ribbon-day click opens that date. Week board is the default screen — one line in App.tsx if Forecast should be home instead. Fixtures rebuilt on top of the duplicate-action_id fix. `make check` green, `npm run build` clean; rendering is unverified in a browser (the Chrome extension would not open a tab this session).

## 20:25 — fixtures from the real pipeline; merge with the week board
make_ui_fixtures.py --date 2026-08-03 now cuts leeward.decision.allocate's own list (100 calls) from the 10,000-veteran cohort and rung 0 scores: 76 act-now, station 630 bookings on top. Rahul: test_no_queue_card_line_names_a_diagnosis is xfail(strict) — the real allocator's rationale names the service and embeds the driver, so the board will show it live too; fix in _rationale or in Week.tsx, then drop the marker.

## 20:55 — redesign: one shell, one demo day, seven screens live against the API
Week board rebuilt (ribbon selects a day; one queue grouped by owner; capacity by unit; not-reached chart), Forecast/Action list/Veteran card/Message re-laid on the template grid, Model report screen built from /report with real decision quality (10.7x oldest-first at 40 calls). Every screen opens on landfall via lib/config.ts; api.ts caches responses and loads the week two days at a time. Board no longer renders rationale (test_ui_fixtures scans for it), so the xfail is gone. npm run build clean; make check green.

## 00:25 — docs/ROUND_C.md
Ran the whole stack live at a88f646 and wrote up what it showed, plus twelve prompts (C0–C11) in
priority and cut order. Five findings, all measured: the capacity slider moves the counter 4.4%
(20→80 calls) because 64.6% of total_eha is the free verified-text bucket every strategy gets —
lift is 1.26x as shown, 2.35x on the scarce slots; a constant predictor at the base rate scores
ECE 0.0000 against Leeward's 0.0011–0.0075, and no screen reports discrimination (within-day AUC
is heat 0.728, treatment_gap 0.707); find_out fires on 4 of 6,303 actions because missingness.py
was never built; tiers.py has none of its four hazard-triggered act-now rules and 234 of 255
act_now veterans get no call; and nothing proves the UI boots offline — node_modules and ui/dist
are both gitignored, so a clean clone with wifi off cannot npm ci.
Gate note: `make check` is red on this machine on `test_the_slider_answers_inside_300ms...`
(395–466 ms) — CPU contention, load avg 27.9 from another lane, not a code change. Everything
else passes. The doc is markdown only. Rendering of the seven screens is still unverified in a
browser; the Chrome extension would not open a tab, third session running.

## 00:54 — screens/Report.tsx: the model report, rebuilt on lane/ui
Recovery as a dot-and-whisker (each 90% interval rescaled to one width, the dot is the truth, misses first, need filters), five log-log reliability panels on a shared axis, harm averted as grouped bars against all three baselines, the ablation table and the fairness table. Built on `main` first by mistake — `lane/ui` was 4 commits ahead with the redesign and a tables-only Report — so it was re-done on `lane/ui`: their `getReport`, types, `.pending` blocks and `NEED_SHORT` kept, my fuller `build_report` (reads the real `report/report.json`, falls back to their decision-quality reconstruction) and full fixture replacing the partial one, their report test extended rather than duplicated.
**Verified in a browser at last**: the Chrome extension still would not open a tab (fourth session), so `Google Chrome --headless=new --screenshot --window-size=1500,6600` against the dev server works and is worth keeping. Checked both audit states — with two rows temporarily flagged the banner goes red, flagged rows get a red rail and a pill, and all 27 passing rows still render. `npm run build` clean; lint clean; 835 passed, 1 skipped. Gate note, same as the last entry: `test_the_slider_answers_inside_300ms...` fails at ~440 ms under a concurrent `leeward.model.fit` (load 27); `git diff lane/ui -- leeward/ tests/test_api.py` is empty, so nothing on that path changed.

## 07:56 — the list states its uncertainty; two denominator bugs on the opening screen
The action list now carries a Risk column: posterior mean, the 80% credible interval, and a
forest-style bar on one fixed 0–100% axis so rows compare. The interval was always in the data
and never on screen — `rationale` states it, and `td.wrap .sub`'s two-line clamp cut it at
"(80% interv…". It matters more than it sounds: the median band is 42 points wide (min 6, max
86), so rank 1 reads 55% and is really 12–95%. `lib/risk.ts` parses the two rationale shapes the
decision layer writes (677/685 carry an interval; the 8 that don't are `check_in_call`, which
report epistemic share instead — that is the Find-out tier explaining itself). Rationale prose
left the table for the row `title`; it was redundant with Top driver.
Two real bugs on the week board, both denominators. The OUTAGE chip read `max(outage_frac)`
labelled "% of ZIPs" — on landfall day it printed **80% of ZIPs when 62% of ZIPs were affected**;
80% was how dark the single worst ZIP was. It now uses Forecast.tsx's own line (`>= 0.2`) so both
screens summarise the field the same way, with the peak moved to the hover. And `facilities` is
one row per site **per day**, so station 630 rendered five identical closure cards under five
duplicate React keys; deduped to one card reading "closed Mon 3 Aug – Fri 7 Aug · 5 days".
`lib/types.ts` was missing `FacilityStatus.date`, which `schemas.py` has carried since the week
contract — that drift is what produced the repeat.
"Versus the baselines" was four zero-based bars of identical length (220.0 / 219.4 / 219.3 /
218.3) and a hero reading "▲ 1.00×". **Measured it rather than assumed**: the free verified-text
bucket is 3.1% of the total here, not the 64.6% ROUND_C.md records, and removing it moves the
ratio from 1.003× to 1.003× — so the dilution story is stale and the flat result is real. At
rung 0 the ranking genuinely is ~0.3% ahead of oldest-first; the prior has no fitted signal to
separate them. Bars from zero can't show that and a truncated bar axis would oversell it, so it
is now a dot plot that prints its own axis range with the gap in need-days (−0.56, −0.66, −1.66),
Leeward emphasised. The hero says "+0.56 vs oldest first" instead of a ratio that rounds to a win.
**Not done, deliberately**: the interval does not go on the week board.
`test_week_board_shows_no_name_no_diagnosis_and_no_bare_eha` bans `rationale` in Week.tsx because
that screen hangs in a shared clinical space, and the only path to the interval is through it.
The gate caught that and it was right; reverted. If the board should carry it, `schemas.py` needs
`p_mean`/`p_lo80`/`p_hi80` on `ActionRow` — which is also the fix that would let the baseline
comparison split scarce from free capacity live, which it cannot today.
`make check` GREEN on this tree (venv verified pointing at this worktree, not the primary).
Verified headless at 1600px on the dev server: action list, week board, dot plot.
## 21:50 — dashboard: event panel, 3-tier patient list, Ask Leeward, resource library
New opening screen (src/screens/Dashboard.tsx). Event panel with computed severity, countdown, NY map and an update feed built from real hazard-table transitions; patient list in three tiers with risk-factor and action counts at a glance, hover detail and click-through to the chart; Ask Leeward answering deterministically offline from the forecast, the allocator's list and the library; 12-card resource library by event and topic with sources. Opens on the first alert day. Six separate commits on lane/ui-redesign so any piece can be reverted alone. npm run build clean.

Rahul: tests/test_api.py::test_the_slider_answers_inside_300ms fails on this Windows machine — median 1412 ms at 10k veterans, budget 300 ms (your note says 127 ms on CI). Pre-existing and not UI: my commits touch only ui/. The client already caches per request and allows 30 s, but the slider will feel slow if the demo runs on a machine like this one.

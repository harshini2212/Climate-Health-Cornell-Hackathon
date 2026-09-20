# demo


## 17:05 — ingest/hazards.py + scenarios/{sandy_then_heat,ida_flash_flood,smoke_2023}.yaml
178 modzcta x 120 days, both tables validate; smoke is an AirNow replay (peak 203.5 on 7 Jun 2023), Sandy closes station 630 for 45 days. 22 tests, ruff + pytest green.

## 19:10 — outreach/verify.py + outreach/messages.py
render(action, veteran) covers all 15 actions and 224/224 real actions render against the 10,000-veteran cohort; both message guardrails now run instead of skip. make check green.

## 20:13 — leeward/demo.py + /forecast default day + the two harm-averted labels
`GET /forecast` with no `day` now answers with the window the demo opens on, which
`leeward/demo.py` reads off the hazards table: two days in front of the first alert-grade
day, never `hot_day` (true from 3 June). For `sandy_then_heat` that is 1–7 Aug — two quiet
days, landfall on the 3rd, the outage, the first two days of the heat wave — instead of 1
June's empty ribbon. Day 0 is still one parameter away: `make demo DAY=0`, the new topbar
picker, or `?day=0`. `make_ui_fixtures.py` defaults to the same rule (`--day` added), so the
offline board opens on the same week the live API does; fixtures rebuilt on 13–19 Jul, that
rule on the fixture calendar. Also fixed on the way past: the nav badge and the un-clicked
action list were posting `date: ""`, which the live API refuses, so both were silently
answering from fixtures — they now post the opening date. 13 new tests; smoke passes offline.

Harm averted is labelled everywhere it appears — "expected · ranking only" on the action
list and forecast tiles, "realized · vs simulated outcomes" on the model report — plus a
checklist bullet in BUILD_PLAN §8. **Partner: the 4.2× in the prompt does not match the
report.** `report/decision_quality.csv` as committed sums to 8.5× at k=40 (10.0× at 20,
7.5× at 80) for leeward vs rank_by_age; 4.2 is a single cell in it (30 Aug, k=80, random).
Quote the ratio off the CSV on the day, not from the prompt. Separately, `leeward` scores
0.0 with `n_veterans=0` on 9 of the 30 held-out days — worth a look before it is on a slide.

**Gate note, not mine:** `test_the_slider_answers_inside_300ms_at_ten_thousand_veterans`
fails intermittently on **`main` as well as here** — median ~300-306 ms against a 300 ms
budget, where status/api.md recorded 127 ms before `allocate.headline` landed. Everything
else is green on a clean checkout of this branch. The slider is the demo's best three
seconds, so the api lane should look at `_headline` before rehearsal.

## 02:13 — ui/dist committed and served by the API; `make clean-clone` now boots and fetches the UI
**Clean clone → offline install → boot → `GET /` (has `id="root"`) + its script + `/api/forecast` in 10.9 s, budget 60** (quiet machine, load 4.8: clone 1.2 · install 1.1 · fixtures 6.1 · boot 2.3 · page+route 0.2; four earlier runs at load 31–57 took 12.6–13.9 s). Network blocked by proxy env vars and `uv --offline`, **not by switching the wifi off** (that would cut this session), and the install came out of a warm uv cache, so a machine that has never run `make setup` still needs one online install; do the wifi-off run once yourself.
`make ui` rebuilds and stamps `ui/dist`; `scripts/ui_dist.py check` compares by content and runs from `make demo`, `make clean-clone` and the gate. Both failures verified loud in under 2 s (stale `ui/src` names the file; missing dist says so). `?day=N` now works at runtime (the entry above claimed it, nothing read it). `make check` is green at the real 300 ms budget on the merged tree once load fell to ~3; while load was 30–60 (another session's model fit) the slider test was red, as on untouched `main` (A/B of this change: 0.994x).

# demo


## 17:05 — ingest/hazards.py + scenarios/{sandy_then_heat,ida_flash_flood,smoke_2023}.yaml
178 modzcta x 120 days, both tables validate; smoke is an AirNow replay (peak 203.5 on 7 Jun 2023), Sandy closes station 630 for 45 days. 22 tests, ruff + pytest green.

## 19:10 — outreach/verify.py + outreach/messages.py
render(action, veteran) covers all 15 actions and 224/224 real actions render against the 10,000-veteran cohort; both message guardrails now run instead of skip. make check green.

# Leeward UI

Vite + React 18 + TypeScript + deck.gl. No UI kit. The look is a port of the Brexify
command-center template (https://brexify.up.railway.app/): its stylesheet, shell and
components, re-expressed for a care team instead of a finance team.

```bash
cd ui
npm install
npm run dev        # http://127.0.0.1:5173, proxies /api to the FastAPI app on :8000
npm run build      # tsc --noEmit && vite build; must be clean before merging
```

Run the API next to it for live numbers; without it every screen falls back to the
committed fixtures and the topbar pill says so:

```bash
python -m uvicorn leeward.api.main:app --port 8000
```

## The demo day

Every screen opens on the same morning. `src/lib/config.ts` holds the scenario and its
demo day index (`day: 63` for Sandy-then-heat, which is landfall, 3 August); `/forecast`
is asked for that day and its first date becomes "today" for the ribbon, the map, the
action list and the card. Opening on day 0 shows an empty ribbon, so nothing does.

## Screens, in demo order

| Screen | What it answers | Data |
| --- | --- | --- |
| **Week board** (default) | What does the week look like, who do I reach first | `/forecast`, `/actions` ×7 at real capacity, ×7 uncapped for "not reached" |
| **Forecast** | The seven-day hazard picture: stat strip, heat and PM2.5 sparks, hazard table, "Needs attention", the morning brief | `/forecast`, `/actions` |
| **Map** | Expected need by ZIP, the 14 VA sites, flood and outage overlays | `/forecast`, `/scores` |
| **Action list** | The ranked list cut at capacity, with the slider and the baselines | `/actions`, `/export` |
| **Veteran card** | Five needs with their uncertainty, drivers, context, medication, the plan, the message | `/veteran/{id}`, `/message/{id}` |
| **Message** | The verified text with its five-element checklist | `/message/{id}` |
| **Model report** | Decision quality (real), and honest "not run yet" cards for calibration, recovery, fairness | `/report` |

Design rules the screens share: one page header, a five-column stat strip, a
`g2` card grid (1.6 : 1), the same badges for tiers everywhere, and no raw enum on
screen. Nominal comparisons (Leeward vs. baselines) use one hue with the lead row set
apart by weight; magnitude on the map uses the sequential blue ramp; status colours are
reserved for hazards and closures and always carry a word.

## De-identification

The week board hangs in a shared clinical space, so it shows a handle (`D.C. · 9639`),
the action, the owner and the tier, and nothing else. Names, conditions, driver phrases
and the allocator's rationale sentence appear only on the veteran card, behind a click.
`tests/test_ui_fixtures.py` scans `Week.tsx` for the fields it must not render.

## Where the numbers come from

Every request goes through `src/lib/api.ts`. It tries the FastAPI app first and falls
back to the committed fixtures in `public/fixtures/`. Responses are cached per request,
so going back to a screen never re-runs an allocation. The week loads today's list first,
then the other six days two at a time, because the API allocates one day at a time and a
10,000-veteran day takes about half a second.

The fixtures are typed against `leeward/api/schemas.py`, and `tests/test_ui_fixtures.py`
parses every one of them through the pydantic models. Regenerate from the real pipeline:

```bash
python -m leeward.ingest.hazards --scenario scenarios/sandy_then_heat.yaml
python -m leeward.cohort.build && python -m leeward.cohort.simulate
python -m leeward.model.score_prior && python -m leeward.decision.allocate
python -m leeward.eval.decision_quality
python scripts/make_ui_fixtures.py --date 2026-08-03
```

`actions_candidates.json` is the real allocator's list at 100 calls; offline the client
cuts it at whatever capacity the slider asks for with the allocator's own rules.
`report.json` (and `report/report.json` for the API) carry decision quality until
`leeward/eval/report.py` assembles the full report.

## Two numbers that look like they disagree

The action list's "▲ 1.13×" is *expected* harm averted on the day, from the posterior.
The model report's "10.7×" is *realised* harm averted on the held-out window, from the
planted outcomes. Both are correct, both are labelled, and the report says so in a line.

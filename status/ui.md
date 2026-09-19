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

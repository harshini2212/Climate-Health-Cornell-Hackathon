# ui


## 17:40 — ui/ scaffold + screens/Map.tsx + screens/CareTeam.tsx + components/CapacitySlider.tsx
Vite+React18+TS+deck.gl; both screens render from public/fixtures (typed against api/schemas.py, tests/test_ui_fixtures.py); slider round-trip ~90 ms; npm run build clean; make check green. ZipHazard has no floodnet_trip field, so the FloodNet toggle is an outage overlay until the api lane adds it.

## 19:20 — screens/Week.tsx is the default screen + components/VeteranCard.tsx + components/Message.tsx
Week ribbon, banners, owner lanes and a per-day queue, de-identified to an initial-plus-id handle with names/conditions/drivers behind a click; day click opens CareTeam on that date (slider kept), card click opens the card and its message checklist. `npm run build` clean, `make check` green.
Two things to know: `ActionsRequest` is single-day so a week is 7 parallel `postActions` (plus 7 more at uncapped capacity — "not reached" cannot be read off one response), and **`/forecast` carries `FacilityStatus` without a date**, so a closure shows as a window-level strip, not on the day it starts. If the api lane adds a per-day site field the ribbon can place it properly. New fixtures `veterans.json` + `messages.json` (from the same `make_ui_fixtures.py` run) make both reveals work offline; offline the queue is one modelled day repeated and the board says so on screen.

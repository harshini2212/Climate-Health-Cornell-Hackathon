# ui


## 17:40 — ui/ scaffold + screens/Map.tsx + screens/CareTeam.tsx + components/CapacitySlider.tsx
Vite+React18+TS+deck.gl; both screens render from public/fixtures (typed against api/schemas.py, tests/test_ui_fixtures.py); slider round-trip ~90 ms; npm run build clean; make check green. ZipHazard has no floodnet_trip field, so the FloodNet toggle is an outage overlay until the api lane adds it.

## 19:30 — design/template.css + App shell + screens/Forecast.tsx; Map and CareTeam restyled
Ported the Brexify command-center template (sidebar, topbar, stat strip, cards, insights, brief, tables) into ui/; new Forecast home from forecast.json + /actions; Inter bundled offline. npm run build clean; make check green. Also made test file reads explicit UTF-8 so the gate passes on Windows.

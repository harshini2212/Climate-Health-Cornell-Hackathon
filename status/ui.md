# ui


## 17:40 — ui/ scaffold + screens/Map.tsx + screens/CareTeam.tsx + components/CapacitySlider.tsx
Vite+React18+TS+deck.gl; both screens render from public/fixtures (typed against api/schemas.py, tests/test_ui_fixtures.py); slider round-trip ~90 ms; npm run build clean; make check green. ZipHazard has no floodnet_trip field, so the FloodNet toggle is an outage overlay until the api lane adds it.

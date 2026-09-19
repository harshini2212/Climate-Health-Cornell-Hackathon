# Leeward UI

Vite + React 18 + TypeScript + deck.gl. No UI kit. Two screens so far: **Map** and **Care team**.

```bash
cd ui
npm install
npm run dev        # http://127.0.0.1:5173
npm run build      # tsc --noEmit && vite build; must be clean before merging
```

## Where the numbers come from

Every request goes through `src/lib/api.ts`. It tries the FastAPI app first (proxied at
`/api` in dev, target `http://127.0.0.1:8000`; set `VITE_API_URL` to point elsewhere) and
falls back to the committed fixtures in `public/fixtures/` when the API is not running.
The badge in the top bar says which one you are looking at.

The fixtures are typed against `leeward/api/schemas.py`, and `tests/test_ui_fixtures.py`
parses every one of them through the pydantic models, so the UI cannot drift from the
contract. Regenerate them with:

```bash
python scripts/make_fixtures.py --start 2026-06-01 --days 120
python -m leeward.ingest.hazards --scenario scenarios/sandy_then_heat.yaml
python scripts/make_ui_fixtures.py
```

`actions_candidates.json` is a ranked *candidate* list, not a cut list. The client cuts
it at whatever capacity the slider asks for (same rules as the real allocator: per-bucket
capacity, one action per veteran unless Act-now, dense ranks by EHA) and computes the
three baselines from the same list. So the capacity slider tells the real story before
`leeward/decision/allocate.py` lands, and switches to the API's answer the moment it does.

## Map

- `GeoJsonLayer` over `data/reference/nyc_modzcta.geojson`, imported as a Vite asset so
  there is one copy of the 178 polygons in git. No basemap tiles: the demo must work with
  the network off.
- Fill is the expected need count per ZIP for the selected need and day, on the
  sequential blue ramp. Gray means nobody on the panel lives there.
- Overlays (outlines): flood warning or surge in status-critical red; power outage above
  20 percent in status-serious orange.
- `IconLayer` of the 14 NYC VA facilities; a site turns red and grows when `site_down`.
  `FacilityStatus` carries no date, so the fixture marks a site down if it is down on
  any day of the 7-day window.

`ZipHazard` (the `/forecast` response shape, owned by the `api` lane) has no
`floodnet_trip` field, so the FloodNet toggle from the spec is not on the map yet; the
outage overlay stands in for it. If the `api` lane adds the field, wire it in `Map.tsx`.

## Care team

`POST /actions` with `{date, capacity}`. Ranked table (rank, name, tier badge, action and
rationale, top driver, EHA, owner), the `CapacitySlider` (10–100, refetches on release),
the harm-averted counter that tweens between values, and the baseline bars.

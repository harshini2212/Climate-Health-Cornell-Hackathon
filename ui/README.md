# Leeward UI

Vite + React 18 + TypeScript + deck.gl. No UI kit. The look is a port of the Brexify
command-center template (https://brexify.up.railway.app/): its stylesheet, shell and
components, re-expressed for a care team instead of a finance team.

```bash
cd ui
npm install
npm run dev        # http://127.0.0.1:5173
npm run build      # tsc --noEmit && vite build; must be clean before merging
```

## The template, and how it maps to Leeward

The reference is one hand-written HTML file: vanilla JS, a 29 KB stylesheet on CSS
variables, Inter, a 230 px sidebar with grouped nav, a sticky topbar, a big page header,
then stat strips, cards, insight tiles, an "AI brief" block and tables. It is not a
published package, so `src/design/template.css` carries the stylesheet with the brand
slots renamed (`--brex` -> `--accent`) and the finance-only pieces dropped. Inter is
bundled from `@fontsource/inter` so the demo stays offline. Light-only, like the source.

| Template screen | Leeward screen | What fills it |
| --- | --- | --- |
| Home, command center | **Forecast** | stat strip (panel, act-now, calls, harm averted, sites down), seven-day hazard charts and table, "Needs attention" insights, care-team morning brief |
| Cards, card portfolio | **Action list** | stat strip by tier, capacity slider, baseline bars, the ranked table with tier filters |
| Research drill-down | **Veteran card** | five needs as interval bars, drivers, medications, plan (lands with the `api` lane) |
| Controls | **Message** | the rendered message with the five mandatory elements as a checklist (`api` lane) |
| Treasury | **Model report** | recovery, calibration, harm averted vs baselines, ablations, fairness (`eval` lane) |
| — | **Map** | deck.gl choropleth in a card, controls in the side column |

Component vocabulary, all from the template: `.strip > .stat` (with `.hero`, `.crit`),
`.card > .ch`, `.row.g2`, `.insights > .insight`, `.aiask` + `.brief`, `.tablewrap > table`,
`.rb` risk badges (tier), `.tag`, `.flagpill`, `.bars > .barrow`, `.prog`, `.ctrls > .iv`.
Leeward-only additions live in `src/design/leeward.css` (map canvas, legend, slider).

## Where the numbers come from

Every request goes through `src/lib/api.ts`. It tries the FastAPI app first (proxied at
`/api` in dev, target `http://127.0.0.1:8000`; set `VITE_API_URL` to point elsewhere) and
falls back to the committed fixtures in `public/fixtures/` when the API is not running.
The pill in the topbar says which one you are looking at.

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
the API's `/actions` is up, and switches to the API's answer the moment it is.

The morning brief on the Forecast screen is written from the numbers on the page with
template strings. It is not a model call and never will be on the demo path.

## Map

- `GeoJsonLayer` over `data/reference/nyc_modzcta.geojson`, imported as a Vite asset so
  there is one copy of the 178 polygons in git. No basemap tiles: the demo must work with
  the network off.
- Fill is the expected need count per ZIP for the selected need and day, on the
  sequential blue ramp. Gray means nobody on the panel lives there.
- Overlays (outlines): flood warning or surge in critical red; power outage above 20
  percent in serious orange.
- `IconLayer` of the 14 NYC VA facilities; a site turns red and grows when `site_down`.
  `FacilityStatus` carries no date, so the fixture marks a site down if it is down on
  any day of the 7-day window.

`ZipHazard` (the `/forecast` response shape, owned by the `api` lane) has no
`floodnet_trip` field, so the FloodNet toggle from the spec is not on the map yet; the
outage overlay stands in for it. If the `api` lane adds the field, wire it in `Map.tsx`.

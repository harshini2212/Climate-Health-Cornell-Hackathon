/**
 * The choropleth, shared by the Map screen and the week board's mini-map. Expected need
 * count per ZIP on the sequential blue ramp, VA facilities as pins (red when down), flood
 * and outage outlines in status colours. No basemap tiles: the demo runs with the wifi off.
 */

import { useEffect, useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { MapView, type PickingInfo } from "@deck.gl/core";
import { GeoJsonLayer, IconLayer } from "@deck.gl/layers";
import geoUrl from "../../../data/reference/nyc_modzcta.geojson?url";
import { getScores } from "../lib/api";
import { FACILITY_DOWN, FACILITY_OPEN, NO_DATA, SEQUENTIAL, SEQUENTIAL_HEX, STATUS, rampIndex, type RGBA } from "../lib/colors";
import { NEED_LABEL, RUNG_LABEL, fmtDate } from "../lib/labels";
import type { FacilityStatus, ForecastResponse, ScoresResponse, ZipHazard } from "../lib/types";

const PIN_SVG =
  `<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">` +
  `<path fill="#000" fill-rule="evenodd" d="M64 4C36 4 16 26 16 52c0 34 48 72 48 72s48-38 48-72C112 26 92 4 64 4z` +
  `m0 30a18 18 0 1 1 0 36 18 18 0 1 1 0-36z"/></svg>`;
const ICON_ATLAS = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(PIN_SVG)}`;
const ICON_MAPPING = { pin: { x: 0, y: 0, width: 128, height: 128, anchorY: 128, mask: true } };

interface Feature {
  properties: { modzcta: string; label: string; pop_est: number };
}

interface Props {
  forecast: ForecastResponse | null;
  date: string | null;
  need: string;
  showFlood?: boolean;
  showOutage?: boolean;
  /** Compact: smaller pins, tighter zoom, legend in the corner without the rung line. */
  compact?: boolean;
  onScores?: (s: ScoresResponse) => void;
}

export function NeedMap({ forecast, date, need, showFlood = true, showOutage = true, compact = false, onScores }: Props) {
  const [scores, setScores] = useState<ScoresResponse | null>(null);

  useEffect(() => {
    if (!date) return;
    let live = true;
    getScores(date, need).then((s) => {
      if (!live) return;
      setScores(s);
      onScores?.(s);
    }).catch(() => undefined);
    return () => {
      live = false;
    };
  }, [date, need, onScores]);

  const hazardByZip = useMemo(() => {
    const m = new globalThis.Map<string, ZipHazard>();
    if (forecast && date) for (const z of forecast.zips) if (z.date === date) m.set(z.modzcta, z);
    return m;
  }, [forecast, date]);

  const scoreByZip = useMemo(() => {
    const m = new globalThis.Map<string, { expected_count: number; lo80: number; hi80: number; n_panel: number }>();
    if (scores) for (const z of scores.zips) m.set(z.modzcta, z);
    return m;
  }, [scores]);

  const maxCount = useMemo(() => {
    let mx = 0;
    scoreByZip.forEach((v) => (mx = Math.max(mx, v.expected_count)));
    return mx;
  }, [scoreByZip]);

  const layers = useMemo(() => {
    const fill = new GeoJsonLayer<Feature["properties"]>({
      id: "zips",
      data: geoUrl,
      filled: true,
      stroked: true,
      getFillColor: (f) => {
        const s = scoreByZip.get(f.properties.modzcta);
        return s ? SEQUENTIAL[rampIndex(s.expected_count, maxCount)] : NO_DATA;
      },
      getLineColor: [255, 255, 255, 255],
      lineWidthMinPixels: 0.6,
      pickable: true,
      autoHighlight: true,
      highlightColor: [25, 25, 25, 60],
      transitions: { getFillColor: 600 },
      updateTriggers: { getFillColor: [scoreByZip, maxCount] },
    });
    const overlay = new GeoJsonLayer<Feature["properties"]>({
      id: "hazard-overlay",
      data: geoUrl,
      filled: false,
      stroked: true,
      getLineColor: (f): RGBA => {
        const h = hazardByZip.get(f.properties.modzcta);
        if (!h) return [0, 0, 0, 0];
        if (showFlood && (h.flood_warning || h.flash_flood_emergency)) return STATUS.critical;
        if (showOutage && h.outage_frac >= 0.2) return STATUS.serious;
        return [0, 0, 0, 0];
      },
      getLineWidth: (f) => {
        const h = hazardByZip.get(f.properties.modzcta);
        if (!h) return 0;
        return (showFlood && (h.flood_warning || h.flash_flood_emergency)) || (showOutage && h.outage_frac >= 0.2) ? (compact ? 1.8 : 2.5) : 0;
      },
      lineWidthUnits: "pixels",
      updateTriggers: { getLineColor: [hazardByZip, showFlood, showOutage], getLineWidth: [hazardByZip, showFlood, showOutage, compact] },
    });
    const facilities = new IconLayer<FacilityStatus>({
      id: "facilities",
      data: forecast?.facilities ?? [],
      iconAtlas: ICON_ATLAS,
      iconMapping: ICON_MAPPING,
      getIcon: () => "pin",
      getPosition: (d) => [d.lon, d.lat],
      getSize: (d) => (compact ? (d.site_down ? 30 : d.site_dependent_services ? 20 : 14) : d.site_down ? 40 : d.site_dependent_services ? 30 : 22),
      sizeUnits: "pixels",
      getColor: (d) => (d.site_down ? FACILITY_DOWN : FACILITY_OPEN),
      pickable: true,
      updateTriggers: { getColor: [forecast], getSize: [forecast, compact] },
    });
    return [fill, overlay, facilities];
  }, [scoreByZip, maxCount, hazardByZip, showFlood, showOutage, forecast, compact]);

  const tooltip = (info: PickingInfo) => {
    const o = info.object as Feature | FacilityStatus | undefined;
    if (!o) return null;
    const style = { background: "none", padding: "0" };
    if ("facility_id" in o) {
      return {
        html:
          `<div class="tooltip"><strong>${o.name}</strong><br/>station ${o.facility_id} · evac zone ${o.evac_zone}` +
          `<br/>${o.site_down ? "<span style='color:var(--crit);font-weight:600'>SITE DOWN</span>" : "open"}` +
          `${o.site_dependent_services ? " · dialysis / infusion / OTP on site" : ""}</div>`,
        style,
      };
    }
    const p = o.properties;
    const s = scoreByZip.get(p.modzcta);
    const h = hazardByZip.get(p.modzcta);
    const flags = h
      ? [h.heat_alert && "heat alert", h.smoke_alert && "smoke", h.flood_warning && "flood warning", h.flash_flood_emergency && "flash flood emergency", h.outage_frac >= 0.2 && `${Math.round(h.outage_frac * 100)}% outage`, h.mail_delivery_disrupted && "mail disrupted"].filter(Boolean)
      : [];
    return {
      html:
        `<div class="tooltip"><strong>${p.label}</strong>` +
        (s ? `<br/>${NEED_LABEL[need]}: <strong>${s.expected_count.toFixed(1)}</strong> expected (${s.lo80.toFixed(1)}–${s.hi80.toFixed(1)}) of ${s.n_panel} on the panel` : "<br/>no veterans on the panel here") +
        (h ? `<br/>heat index ${h.heat_index_max_f.toFixed(0)}°F · PM2.5 ${h.pm25.toFixed(0)}` : "") +
        (flags.length ? `<br/><span style='color:var(--amber)'>${flags.join(" · ")}</span>` : "") +
        `</div>`,
      style,
    };
  };

  const view = compact
    ? { longitude: -73.92, latitude: 40.70, zoom: 8.9, minZoom: 8, maxZoom: 13 }
    : { longitude: -73.93, latitude: 40.705, zoom: 9.4, minZoom: 8, maxZoom: 14 };

  return (
    <>
      <DeckGL views={new MapView({ id: "map", repeat: false })} initialViewState={view} controller={true} layers={layers} getTooltip={tooltip} style={{ position: "absolute", inset: "0" }} />
      <div className={`maplegend${compact ? " compact" : ""}`}>
        <div style={{ fontWeight: 600 }}>{NEED_LABEL[need]}</div>
        <div className="muted">expected count · {date ? fmtDate(date) : ""}</div>
        <div className="ramp">{SEQUENTIAL_HEX.map((h) => <span key={h} style={{ background: h }} />)}</div>
        <div className="ends"><span>0</span><span>{maxCount.toFixed(1)}</span></div>
        {!compact && scores && <div className="muted" style={{ marginTop: 4 }}>{RUNG_LABEL[scores.model_rung]}</div>}
      </div>
    </>
  );
}

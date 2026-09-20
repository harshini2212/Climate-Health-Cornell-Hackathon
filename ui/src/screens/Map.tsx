import { useEffect, useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { MapView, type PickingInfo } from "@deck.gl/core";
import { GeoJsonLayer, IconLayer } from "@deck.gl/layers";
import geoUrl from "../../../data/reference/nyc_modzcta.geojson?url";
import { IconAlert } from "../components/Icons";
import { getForecast, getScores, lastSource, type Source } from "../lib/api";
import { FACILITY_DOWN, FACILITY_OPEN, NO_DATA, SEQUENTIAL, SEQUENTIAL_HEX, STATUS, STATUS_HEX, rampIndex, type RGBA } from "../lib/colors";
import { NEED_LABEL, RUNG_LABEL, fmtDate } from "../lib/labels";
import { NEEDS, type FacilityStatus, type ForecastResponse, type ScoresResponse, type ZipHazard } from "../lib/types";

// A pin with a transparent hole. `mask: true` means only its alpha is used and the
// color comes from getColor, so one atlas serves open (blue) and down (red) sites.
const PIN_SVG =
  `<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">` +
  `<path fill="#000" fill-rule="evenodd" d="M64 4C36 4 16 26 16 52c0 34 48 72 48 72s48-38 48-72C112 26 92 4 64 4z` +
  `m0 30a18 18 0 1 1 0 36 18 18 0 1 1 0-36z"/></svg>`;
const ICON_ATLAS = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(PIN_SVG)}`;
const ICON_MAPPING = { pin: { x: 0, y: 0, width: 128, height: 128, anchorY: 128, mask: true } };
const INITIAL_VIEW = { longitude: -73.93, latitude: 40.705, zoom: 9.4, minZoom: 8, maxZoom: 14 };

interface Feature {
  properties: { modzcta: string; label: string; pop_est: number };
}

export function Map({ scenario, day, onSource }: { scenario: string; day?: number; onSource: (s: Source) => void }) {
  const [need, setNeed] = useState<string>("treatment_gap");
  const [dayIdx, setDayIdx] = useState(0);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [scores, setScores] = useState<ScoresResponse | null>(null);
  const [showFlood, setShowFlood] = useState(true);
  const [showOutage, setShowOutage] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getForecast(scenario, day).then((f) => { setForecast(f); setDayIdx(0); onSource(lastSource()); }).catch((e) => setError(String(e)));
  }, [scenario, day, onSource]);

  const date = forecast?.dates[dayIdx] ?? null;
  useEffect(() => {
    if (!date) return;
    getScores(date, need).then((s) => { setScores(s); onSource(lastSource()); }).catch((e) => setError(String(e)));
  }, [date, need, onSource]);

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
        return (showFlood && (h.flood_warning || h.flash_flood_emergency)) || (showOutage && h.outage_frac >= 0.2) ? 2.5 : 0;
      },
      lineWidthUnits: "pixels",
      updateTriggers: { getLineColor: [hazardByZip, showFlood, showOutage], getLineWidth: [hazardByZip, showFlood, showOutage] },
    });
    const facilities = new IconLayer<FacilityStatus>({
      id: "facilities",
      data: forecast?.facilities ?? [],
      iconAtlas: ICON_ATLAS,
      iconMapping: ICON_MAPPING,
      getIcon: () => "pin",
      getPosition: (d) => [d.lon, d.lat],
      getSize: (d) => (d.site_down ? 40 : d.site_dependent_services ? 30 : 22),
      sizeUnits: "pixels",
      getColor: (d) => (d.site_down ? FACILITY_DOWN : FACILITY_OPEN),
      pickable: true,
      updateTriggers: { getColor: [forecast], getSize: [forecast] },
    });
    return [fill, overlay, facilities];
  }, [scoreByZip, maxCount, hazardByZip, showFlood, showOutage, forecast]);

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

  const majors = forecast?.facilities.filter((f) => f.site_dependent_services || f.site_down) ?? [];
  const down = forecast?.facilities.filter((f) => f.site_down) ?? [];

  if (error) return <div className="page"><div className="empty"><div className="eh">Could not load</div>{error}</div></div>;

  return (
    <div className="page dash">
      {forecast?.headline && (
        <div className="insight">
          <div className="ic ic-medium"><IconAlert /></div>
          <div className="bd"><div className="t">{forecast.headline}</div><div className="d">{forecast.dates.length}-day window from {fmtDate(forecast.dates[0])}</div></div>
        </div>
      )}
      <div className="row g2">
        <div className="card tight">
          <div className="mapcanvas">
            <DeckGL views={new MapView({ id: "map", repeat: false })} initialViewState={INITIAL_VIEW} controller={true} layers={layers} getTooltip={tooltip} style={{ position: "absolute", inset: "0" }} />
            <div className="maplegend">
              <div style={{ fontWeight: 600 }}>{NEED_LABEL[need]}</div>
              <div className="muted">expected count · {date ? fmtDate(date) : ""}</div>
              <div className="ramp">{SEQUENTIAL_HEX.map((h) => <span key={h} style={{ background: h }} />)}</div>
              <div className="ends"><span>0</span><span>{maxCount.toFixed(1)}</span></div>
              {scores && <div className="muted" style={{ marginTop: 4 }}>{RUNG_LABEL[scores.model_rung]}</div>}
            </div>
          </div>
        </div>

        <div className="stack">
          <div className="card">
            <div className="ch"><h3>Need and day</h3></div>
            <select value={need} onChange={(e) => setNeed(e.target.value)} aria-label="Need" style={{ width: "100%" }}>
              {NEEDS.map((n) => <option key={n} value={n}>{NEED_LABEL[n]}</option>)}
            </select>
            <div className="slider" style={{ marginTop: 16 }}>
              <div className="top"><span className="muted">Day</span><span style={{ fontWeight: 600 }}>{date ? fmtDate(date) : "—"}</span></div>
              <input type="range" min={0} max={Math.max(0, (forecast?.dates.length ?? 1) - 1)} value={dayIdx} onChange={(e) => setDayIdx(Number(e.target.value))} aria-label="Forecast day" />
              <div className="ticks"><span>{forecast ? fmtDate(forecast.dates[0]) : ""}</span><span>{forecast ? fmtDate(forecast.dates[forecast.dates.length - 1]) : ""}</span></div>
            </div>
          </div>

          <div className="card">
            <div className="ch"><h3>Overlays</h3></div>
            <label className="check"><input type="checkbox" checked={showFlood} onChange={(e) => setShowFlood(e.target.checked)} /><span className="swatch" style={{ background: STATUS_HEX.critical }} /> Flood warning or surge</label>
            <label className="check"><input type="checkbox" checked={showOutage} onChange={(e) => setShowOutage(e.target.checked)} /><span className="swatch" style={{ background: STATUS_HEX.serious }} /> Power outage ≥ 20 %</label>
          </div>

          <div className="card">
            <div className="ch"><h3>VA facilities</h3><span className="sp" />{down.length ? <span className="pillbadge pill-critical">{down.length} down</span> : <span className="pillbadge pill-healthy">all open</span>}</div>
            <div className="dleg"><i style={{ background: "#2a78d6" }} /> open</div>
            <div className="dleg" style={{ marginBottom: 8 }}><i style={{ background: STATUS_HEX.critical }} /> site down in the window</div>
            {majors.map((f) => (
              <div key={f.facility_id} className={`facrow${f.site_down ? " down" : ""}`}>
                <span className="nm" title={f.name}>{f.facility_id} · {f.name}</span>
                {f.site_down ? <span className="rb rb-critical">DOWN</span> : <span className="tag">zone {f.evac_zone}</span>}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

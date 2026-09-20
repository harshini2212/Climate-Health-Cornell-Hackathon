import { useEffect, useState } from "react";
import { IconAlert } from "../components/Icons";
import { NeedMap } from "../components/NeedMap";
import { getForecast, lastSource, type Source } from "../lib/api";
import { STATUS_HEX } from "../lib/colors";
import { NEED_LABEL, fmtDate } from "../lib/labels";
import { NEEDS, type ForecastResponse } from "../lib/types";

export function Map({ scenario, day, onSource }: { scenario: string; day?: number; onSource: (s: Source) => void }) {
  const [need, setNeed] = useState<string>("treatment_gap");
  const [dayIdx, setDayIdx] = useState(0);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [showFlood, setShowFlood] = useState(true);
  const [showOutage, setShowOutage] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getForecast(scenario, day).then((f) => { setForecast(f); setDayIdx(0); onSource(lastSource()); }).catch((e) => setError(String(e)));
  }, [scenario, day, onSource]);

  const date = forecast?.dates[dayIdx] ?? null;
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
            <NeedMap forecast={forecast} date={date} need={need} showFlood={showFlood} showOutage={showOutage} />
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

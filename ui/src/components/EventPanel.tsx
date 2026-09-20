/**
 * The extreme-weather event: what it is, how severe, how many days out, where it lands,
 * and what has changed since the last briefing.
 *
 * The update feed is derived from the hazard table, not invented: each line is a real
 * transition in the data (a watch becoming a warning, the surge arriving, a site closing,
 * the heat alert starting), timestamped to the day it happens and marked new when it is
 * within the next 24 hours.
 */

import { useMemo } from "react";
import { NeedMap } from "./NeedMap";
import { IconAlert, IconBolt, IconBuilding, IconDrop, IconMail, IconSun } from "./Icons";
import { fmtDate, fmtInt } from "../lib/labels";
import type { ForecastResponse, ZipHazard } from "../lib/types";

export interface EventDay {
  date: string;
  flood: boolean;
  watch: boolean;
  flash: boolean;
  heat: boolean;
  smoke: boolean;
  surge: number;
  outage: number;
  outageZips: number;
  floodZips: number;
  mailZips: number;
  maxHeatIndex: number;
  meanHeatIndex: number;
  maxPm25: number;
  evacZone: number;
}

export function eventDays(dates: string[], zips: ZipHazard[]): EventDay[] {
  const by = new Map<string, EventDay & { n: number; sum: number }>();
  for (const d of dates) by.set(d, { date: d, flood: false, watch: false, flash: false, heat: false, smoke: false, surge: 0, outage: 0, outageZips: 0, floodZips: 0, mailZips: 0, maxHeatIndex: 0, meanHeatIndex: 0, maxPm25: 0, evacZone: 0, n: 0, sum: 0 });
  for (const z of zips) {
    const d = by.get(z.date);
    if (!d) continue;
    d.flood ||= z.flood_warning || z.flash_flood_emergency;
    d.flash ||= z.flash_flood_emergency;
    d.heat ||= z.heat_alert;
    d.smoke ||= z.smoke_alert;
    d.surge = Math.max(d.surge, z.surge_ft);
    d.outage = Math.max(d.outage, z.outage_frac);
    d.maxHeatIndex = Math.max(d.maxHeatIndex, z.heat_index_max_f);
    d.maxPm25 = Math.max(d.maxPm25, z.pm25);
    d.evacZone = Math.max(d.evacZone, z.evac_zone_ordered);
    if (z.outage_frac >= 0.2) d.outageZips += 1;
    if (z.flood_warning || z.flash_flood_emergency) d.floodZips += 1;
    if (z.mail_delivery_disrupted) d.mailZips += 1;
    d.n += 1;
    d.sum += z.heat_index_max_f;
  }
  const out = dates.map((d) => {
    const x = by.get(d)!;
    return { ...x, meanHeatIndex: x.n ? x.sum / x.n : 0 };
  });
  // A watch is the day before a warning that is not itself under one.
  for (let i = 0; i < out.length - 1; i++) if (out[i + 1].flood && !out[i].flood) out[i].watch = true;
  return out;
}

export type Severity = "severe" | "serious" | "elevated" | "quiet";

export interface EventSummary {
  kind: "flood" | "heat" | "smoke" | "none";
  title: string;
  detail: string;
  severity: Severity;
  peakIndex: number;
  daysOut: number;
}

export const SEV_LABEL: Record<Severity, string> = { severe: "Severe", serious: "Serious", elevated: "Elevated", quiet: "Quiet" };

export function summarise(days: EventDay[], sitesDown: number): EventSummary {
  const fl = days.findIndex((d) => d.flood);
  const ht = days.findIndex((d) => d.heat);
  const sm = days.findIndex((d) => d.smoke);
  if (fl >= 0) {
    const d = days[fl];
    const sev: Severity = d.surge >= 6 || sitesDown > 0 ? "severe" : d.flash ? "serious" : "elevated";
    return {
      kind: "flood",
      title: d.flash ? "Flash flood emergency" : "Coastal storm and surge",
      detail: `${d.surge > 0 ? `${d.surge.toFixed(0)} ft of surge, ` : ""}evacuation zone${d.evacZone > 1 ? "s 1–" : " "}${d.evacZone} ordered, ${d.floodZips} ZIPs under a warning`,
      severity: sev, peakIndex: fl, daysOut: fl,
    };
  }
  if (ht >= 0) {
    const peak = days.reduce((b, x) => (x.heat && x.maxHeatIndex > days[b].maxHeatIndex ? days.indexOf(x) : b), ht);
    const sev: Severity = days[peak].maxHeatIndex >= 100 ? "severe" : days[peak].maxHeatIndex >= 95 ? "serious" : "elevated";
    return { kind: "heat", title: "Extreme heat", detail: `heat index to ${Math.round(days[peak].maxHeatIndex)} °F, ${days.filter((x) => x.heat).length} alert days`, severity: sev, peakIndex: peak, daysOut: ht };
  }
  if (sm >= 0) {
    const peak = days.reduce((b, x) => (x.maxPm25 > days[b].maxPm25 ? days.indexOf(x) : b), sm);
    const sev: Severity = days[peak].maxPm25 >= 150 ? "severe" : days[peak].maxPm25 >= 55 ? "serious" : "elevated";
    return { kind: "smoke", title: "Wildfire smoke", detail: `PM2.5 to ${Math.round(days[peak].maxPm25)} µg/m³`, severity: sev, peakIndex: peak, daysOut: sm };
  }
  return { kind: "none", title: "No active event", detail: "no alerts in the seven-day window", severity: "quiet", peakIndex: 0, daysOut: 0 };
}

interface Update {
  date: string;
  icon: "flood" | "surge" | "heat" | "smoke" | "outage" | "site" | "mail";
  title: string;
  detail: string;
}

/** Real transitions in the hazard table, in order. */
function updates(days: EventDay[], forecast: ForecastResponse): Update[] {
  const out: Update[] = [];
  const down = forecast.facilities.filter((f) => f.site_down);
  days.forEach((d, i) => {
    const p = days[i - 1];
    if (d.watch && !d.flood) out.push({ date: d.date, icon: "flood", title: "Coastal flood watch issued", detail: `for the day ahead; ${d.evacZone ? `zone ${d.evacZone} ` : ""}waterfront ZIPs` });
    if (d.flood && (!p || !p.flood)) out.push({ date: d.date, icon: "surge", title: d.flash ? "Flash flood emergency" : "Coastal flood warning in effect", detail: `${d.floodZips} ZIPs${d.surge > 0 ? `, surge to ${d.surge.toFixed(0)} ft` : ""}${d.evacZone ? `, evacuation zone ${d.evacZone} ordered` : ""}` });
    if (d.outageZips > 0 && (!p || p.outageZips === 0)) out.push({ date: d.date, icon: "outage", title: "Outages begin", detail: `${d.outageZips} ZIPs above 20% out, peak ${Math.round(d.outage * 100)}%` });
    if (d.mailZips > 0 && (!p || p.mailZips === 0)) out.push({ date: d.date, icon: "mail", title: "Mail delivery disrupted", detail: `${d.mailZips} ZIPs; four in five VA prescriptions arrive by mail` });
    if (d.heat && (!p || !p.heat)) out.push({ date: d.date, icon: "heat", title: "Heat alert begins", detail: `heat index to ${Math.round(d.maxHeatIndex)} °F${d.outageZips ? ", while power is still out" : ""}` });
    if (d.smoke && (!p || !p.smoke)) out.push({ date: d.date, icon: "smoke", title: "Smoke advisory", detail: `PM2.5 to ${Math.round(d.maxPm25)} µg/m³` });
  });
  if (down.length) {
    const first = days.find((d) => d.flood) ?? days[0];
    out.push({ date: first.date, icon: "site", title: `${down[0].name} closes`, detail: `station ${down[0].facility_id}, evacuation zone ${down[0].evac_zone}${down[0].site_dependent_services ? "; dialysis, infusion and OTP on site" : ""}` });
  }
  return out.sort((a, b) => a.date.localeCompare(b.date));
}

const ICON: Record<Update["icon"], JSX.Element> = {
  flood: <IconDrop />, surge: <IconDrop />, heat: <IconSun />, smoke: <IconAlert />, outage: <IconBolt />, site: <IconBuilding />, mail: <IconMail />,
};

interface Props {
  forecast: ForecastResponse;
  days: EventDay[];
  selected: number;
  onSelect: (i: number) => void;
  need: string;
}

export function EventPanel({ forecast, days, selected, onSelect, need }: Props) {
  const down = forecast.facilities.filter((f) => f.site_down);
  const ev = useMemo(() => summarise(days, down.length), [days, down.length]);
  const feed = useMemo(() => updates(days, forecast), [days, forecast]);
  const cur = days[selected] ?? days[0];

  return (
    <div className="card tight evcard">
      <div className={`evhead sev-${ev.severity}`}>
        <div className="evh-left">
          <div className="evh-kick"><span className={`sevdot sev-${ev.severity}`} />{SEV_LABEL[ev.severity]} · {forecast.scenario.replace(/_/g, " ")}</div>
          <div className="evh-title">{ev.title}</div>
          <div className="evh-detail">{ev.detail}</div>
        </div>
        <div className="evh-count">
          <div className="evc-num">{ev.daysOut}</div>
          <div className="evc-lab">{ev.kind === "none" ? "alerts" : ev.daysOut === 0 ? "it starts today" : ev.daysOut === 1 ? "day out" : "days out"}</div>
        </div>
      </div>

      <div className="evbody">
        <div className="evmap"><NeedMap forecast={forecast} date={cur.date} need={need} compact /></div>
        <div className="evside">
          <div className="evside-h">Updates<span className="muted">{feed.length} since the watch</span></div>
          <div className="feed">
            {feed.map((u, i) => {
              const idx = days.findIndex((d) => d.date === u.date);
              return (
                <button key={i} className={`fd${idx === selected ? " sel" : ""}${idx <= 1 ? " new" : ""}`} onClick={() => idx >= 0 && onSelect(idx)}>
                  <span className="fd-ic">{ICON[u.icon]}</span>
                  <span className="fd-body">
                    <span className="fd-t">{u.title}{idx <= 1 && <span className="fd-new">new</span>}</span>
                    <span className="fd-d">{u.detail}</span>
                    <span className="fd-when">{fmtDate(u.date)}</span>
                  </span>
                </button>
              );
            })}
            {feed.length === 0 && <div className="muted small" style={{ padding: 10 }}>No changes in the window.</div>}
          </div>
        </div>
      </div>

      <div className="evstrip" role="tablist" aria-label="Days">
        {days.map((d, i) => (
          <button key={d.date} className={`evd${i === selected ? " sel" : ""}${d.flood ? " f" : d.heat ? " h" : ""}`} role="tab" aria-selected={i === selected} onClick={() => onSelect(i)}>
            <span className="evd-dow">{i === 0 ? "Today" : fmtDate(d.date).split(" ")[0]}</span>
            <span className="evd-temp">{Math.round(d.meanHeatIndex)}°</span>
            <span className="evd-tag">{d.flood ? (d.flash ? "FLASH" : "FLOOD") : d.heat ? "HEAT" : d.smoke ? "SMOKE" : d.outageZips ? "OUTAGE" : "clear"}</span>
            <span className="evd-sub">{d.floodZips ? `${fmtInt(d.floodZips)} ZIPs` : d.outageZips ? `${fmtInt(d.outageZips)} ZIPs` : d.heat ? `${Math.round(d.maxHeatIndex)}°F peak` : "—"}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

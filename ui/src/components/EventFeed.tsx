/**
 * One summary of the week's hazards, and the update feed built from it.
 *
 * There used to be two summarisers, one here and one in the week board, which is how the
 * outage chip ended up saying two different things. There is one now, and every screen
 * reads it.
 *
 * The feed is derived, not written: each line is a real transition in the hazard table (a
 * watch becoming a warning, the surge arriving, outages starting, mail disruption, a site
 * closing, the heat alert), on the day it happens.
 */

import { useMemo } from "react";
import { IconAlert, IconBolt, IconBuilding, IconDrop, IconMail, IconSun } from "./Icons";
import { fmtDate } from "../lib/labels";
import type { ForecastResponse, ZipHazard } from "../lib/types";

/**
 * The line Forecast.tsx uses to call a ZIP "in outage" (`outage_frac >= 0.2`). Every
 * screen summarises the same field, so every screen summarises it the same way.
 */
export const OUTAGE_LINE = 0.2;

export interface EventDay {
  date: string;
  flood: boolean;
  /** The day before a warning that is not itself under one. */
  watch: boolean;
  flash: boolean;
  heat: boolean;
  smoke: boolean;
  surge: number;
  /** ZIPs at or over the outage line, and how many ZIPs the day had at all. */
  outageZips: number;
  nZips: number;
  /** Worst single ZIP's outage fraction. Not a share of ZIPs. */
  outagePeak: number;
  floodZips: number;
  mailZips: number;
  maxHeatIndex: number;
  /** Mean over ZIPs: what the timeline prints, so one hot ZIP does not set the day. */
  meanHeatIndex: number;
  maxPm25: number;
  evacZone: number;
}

export function eventDays(dates: string[], zips: ZipHazard[]): EventDay[] {
  const by = new Map<string, EventDay & { sum: number }>();
  for (const d of dates) {
    by.set(d, {
      date: d, flood: false, watch: false, flash: false, heat: false, smoke: false,
      surge: 0, outageZips: 0, nZips: 0, outagePeak: 0, floodZips: 0, mailZips: 0,
      maxHeatIndex: 0, meanHeatIndex: 0, maxPm25: 0, evacZone: 0, sum: 0,
    });
  }
  for (const z of zips) {
    const d = by.get(z.date);
    if (!d) continue;
    d.flood ||= z.flood_warning || z.flash_flood_emergency;
    d.flash ||= z.flash_flood_emergency;
    d.heat ||= z.heat_alert;
    d.smoke ||= z.smoke_alert;
    d.surge = Math.max(d.surge, z.surge_ft);
    d.outagePeak = Math.max(d.outagePeak, z.outage_frac);
    d.maxHeatIndex = Math.max(d.maxHeatIndex, z.heat_index_max_f);
    d.maxPm25 = Math.max(d.maxPm25, z.pm25);
    d.evacZone = Math.max(d.evacZone, z.evac_zone_ordered);
    if (z.outage_frac >= OUTAGE_LINE) d.outageZips += 1;
    if (z.flood_warning || z.flash_flood_emergency) d.floodZips += 1;
    if (z.mail_delivery_disrupted) d.mailZips += 1;
    d.nZips += 1;
    d.sum += z.heat_index_max_f;
  }
  const out = dates.map((d) => {
    const x = by.get(d)!;
    return { ...x, meanHeatIndex: x.nZips ? x.sum / x.nZips : 0 };
  });
  for (let i = 0; i < out.length - 1; i++) if (out[i + 1].flood && !out[i].flood) out[i].watch = true;
  return out;
}

/**
 * Words, not pictograms: a two-metre read beats an icon nobody has learned yet.
 *
 * The outage chip reads the *share of ZIPs*, which is what its label promises. The peak
 * fraction is a different number and lives in the hover, where it matters for the
 * powered-equipment patients.
 */
export function chipsFor(d: EventDay) {
  const out: { key: string; label: string; detail: string; title?: string }[] = [];
  if (d.flood) out.push({ key: "flood", label: "FLOOD", detail: d.evacZone ? `evac zone ${d.evacZone}` : "warning" });
  if (d.surge > 0) out.push({ key: "surge", label: "SURGE", detail: `${d.surge.toFixed(0)} ft` });
  if (d.heat) out.push({ key: "heat", label: "HEAT", detail: `${Math.round(d.maxHeatIndex)}°F index` });
  if (d.smoke) out.push({ key: "smoke", label: "SMOKE", detail: `PM2.5 ${Math.round(d.maxPm25)}` });
  if (d.outageZips > 0) {
    const share = d.nZips ? Math.round((100 * d.outageZips) / d.nZips) : 0;
    out.push({
      key: "outage", label: "OUTAGE", detail: `${share}% of ZIPs`,
      title: `${d.outageZips} of ${d.nZips} ZIPs at or over ${Math.round(OUTAGE_LINE * 100)}% of customers out; worst ZIP ${Math.round(d.outagePeak * 100)}%`,
    });
  }
  return out;
}

export type Severity = "severe" | "serious" | "elevated" | "quiet";

export interface EventSummary {
  kind: "flood" | "heat" | "smoke" | "none";
  title: string;
  sub: string;
  severity: Severity;
  /** Index of the first alert day, which is what the countdown counts to. */
  daysOut: number;
  label: string;
}

const FULL: Record<string, string> = {
  Sun: "Sunday", Mon: "Monday", Tue: "Tuesday", Wed: "Wednesday",
  Thu: "Thursday", Fri: "Friday", Sat: "Saturday",
};

/** The one line the room needs, and how far away it is. */
export function summarise(days: EventDay[], sitesDown: number): EventSummary {
  const when = (i: number) => {
    if (i === 0) return "today";
    if (i === 1) return "tomorrow";
    const short = fmtDate(days[i].date).split(" ")[0];
    return `on ${FULL[short] ?? short}`;
  };
  const fl = days.findIndex((d) => d.flood);
  const ht = days.findIndex((d) => d.heat);
  const sm = days.findIndex((d) => d.smoke);

  if (fl >= 0) {
    const d = days[fl];
    const sev: Severity = d.surge >= 6 || sitesDown > 0 ? "severe" : d.flash ? "serious" : "elevated";
    const sub = ht > fl
      ? `then a heat wave from ${fmtDate(days[ht].date)}, while power is still out`
      : `${d.surge > 0 ? `${d.surge.toFixed(0)} ft of surge, ` : ""}evacuation zones ordered, ${d.floodZips} ZIPs under a warning`;
    return {
      kind: "flood",
      title: d.flash ? `Flash flood emergency ${when(fl)}` : `Coastal storm makes landfall ${when(fl)}`,
      sub, severity: sev, daysOut: fl,
      label: fl === 0 ? "landfall today" : fl === 1 ? "day to landfall" : "days to landfall",
    };
  }
  if (ht >= 0) {
    const peak = days.reduce((b, x, i) => (x.heat && x.maxHeatIndex > days[b].maxHeatIndex ? i : b), ht);
    const sev: Severity = days[peak].maxHeatIndex >= 100 ? "severe" : days[peak].maxHeatIndex >= 95 ? "serious" : "elevated";
    return {
      kind: "heat", title: `Heat wave from ${when(ht)}`,
      sub: `heat index to ${Math.round(days[peak].maxHeatIndex)} °F at the peak, ${days.filter((x) => x.heat).length} alert days`,
      severity: sev, daysOut: ht, label: ht === 1 ? "day to the heat wave" : "days to the heat wave",
    };
  }
  if (sm >= 0) {
    const peak = days.reduce((b, x, i) => (x.maxPm25 > days[b].maxPm25 ? i : b), sm);
    const sev: Severity = days[peak].maxPm25 >= 150 ? "severe" : days[peak].maxPm25 >= 55 ? "serious" : "elevated";
    return {
      kind: "smoke", title: `Wildfire smoke from ${when(sm)}`,
      sub: `PM2.5 to ${Math.round(days[peak].maxPm25)} µg/m³`,
      severity: sev, daysOut: sm, label: "days to the smoke",
    };
  }
  return { kind: "none", title: "A quiet week", sub: "no alerts in the seven-day window", severity: "quiet", daysOut: 0, label: "alerts" };
}

// --------------------------------------------------------------------------- //

interface Update {
  date: string;
  icon: "flood" | "surge" | "heat" | "smoke" | "outage" | "site" | "mail";
  title: string;
  detail: string;
}

export function updatesFor(days: EventDay[], forecast: ForecastResponse): Update[] {
  const out: Update[] = [];
  days.forEach((d, i) => {
    const p = days[i - 1];
    if (d.watch && !d.flood) out.push({ date: d.date, icon: "flood", title: "Coastal flood watch issued", detail: `for the day ahead${d.evacZone ? `, zone ${d.evacZone}` : ""}; waterfront ZIPs` });
    if (d.flood && (!p || !p.flood)) out.push({ date: d.date, icon: "surge", title: d.flash ? "Flash flood emergency" : "Coastal flood warning in effect", detail: `${d.floodZips} ZIPs${d.surge > 0 ? `, surge to ${d.surge.toFixed(0)} ft` : ""}${d.evacZone ? `, evacuation zone ${d.evacZone} ordered` : ""}` });
    if (d.outageZips > 0 && (!p || p.outageZips === 0)) out.push({ date: d.date, icon: "outage", title: "Outages begin", detail: `${d.outageZips} of ${d.nZips} ZIPs out, worst ZIP ${Math.round(d.outagePeak * 100)}%` });
    if (d.mailZips > 0 && (!p || p.mailZips === 0)) out.push({ date: d.date, icon: "mail", title: "Mail delivery disrupted", detail: `${d.mailZips} ZIPs; four in five VA prescriptions arrive by mail` });
    if (d.heat && (!p || !p.heat)) out.push({ date: d.date, icon: "heat", title: "Heat alert begins", detail: `heat index to ${Math.round(d.maxHeatIndex)} °F${d.outageZips ? ", while power is still out" : ""}` });
    if (d.smoke && (!p || !p.smoke)) out.push({ date: d.date, icon: "smoke", title: "Smoke advisory", detail: `PM2.5 to ${Math.round(d.maxPm25)} µg/m³` });
  });
  const seen = new Set<string>();
  for (const f of forecast.facilities) {
    if (!f.site_down || seen.has(f.facility_id)) continue;
    seen.add(f.facility_id);
    const first = days.find((d) => d.flood) ?? days[0];
    out.push({
      date: first.date, icon: "site", title: `${f.name} closes`,
      detail: `station ${f.facility_id}, evacuation zone ${f.evac_zone}${f.site_dependent_services ? "; dialysis, infusion and OTP on site" : ""}`,
    });
  }
  return out.sort((a, b) => a.date.localeCompare(b.date));
}

const ICON: Record<Update["icon"], JSX.Element> = {
  flood: <IconDrop />, surge: <IconDrop />, heat: <IconSun />, smoke: <IconAlert />,
  outage: <IconBolt />, site: <IconBuilding />, mail: <IconMail />,
};

interface FeedProps {
  days: EventDay[];
  forecast: ForecastResponse;
  selected: number;
  onSelect: (i: number) => void;
}

export function UpdateFeed({ days, forecast, selected, onSelect }: FeedProps) {
  const feed = useMemo(() => updatesFor(days, forecast), [days, forecast]);
  if (feed.length === 0) return <div className="muted small" style={{ padding: 10 }}>No changes in the window.</div>;
  return (
    <div className="feed">
      {feed.map((u, i) => {
        const idx = days.findIndex((d) => d.date === u.date);
        return (
          <button key={i} className={`fd${idx === selected ? " sel" : ""}${idx <= 1 ? " new" : ""}`} onClick={() => idx >= 0 && onSelect(idx)} title={`Show ${fmtDate(u.date)}`}>
            <span className="fd-ic">{ICON[u.icon]}</span>
            <span className="fd-body">
              <span className="fd-t">{u.title}{idx <= 1 && <span className="fd-new">new</span>}</span>
              <span className="fd-d">{u.detail}</span>
              <span className="fd-when">{fmtDate(u.date)}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

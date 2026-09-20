/**
 * The dashboard: the screen the care team opens and the one that goes on the big screen.
 *
 * Four things, in the order a team needs them:
 *   1. the event: what it is, how severe, how many days out, where it lands, what changed
 *   2. the at-risk patient list in three tiers, with risk factors and actions at a glance
 *   3. Ask Leeward, answering from the data on screen
 *   4. the resource library, filtered to the event in progress
 *
 * Everything below the event panel follows the day selected in it, so one click moves the
 * whole screen to landfall day or to the first day of the heat wave.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { AskPanel } from "../components/AskPanel";
import { CountUp } from "../components/CountUp";
import { EventPanel, eventDays, summarise } from "../components/EventPanel";
import { IconArrow } from "../components/Icons";
import { PatientList } from "../components/PatientList";
import { ResourceCard } from "./Library";
import { getForecast, lastSource, postActions, postActionsWeek, type Source } from "../lib/api";
import type { AskContext } from "../lib/ask";
import { searchLibrary, type EventKind } from "../lib/library";
import { fmtDate, fmtInt } from "../lib/labels";
import { DEFAULT_CAPACITY, type ActionsResponse, type Capacity, type ForecastResponse } from "../lib/types";
import type { ScreenKey } from "../App";
import type { VeteranFocus } from "./Week";

const UNCAPPED: Capacity = Object.fromEntries(Object.keys(DEFAULT_CAPACITY).map((k) => [k, 100_000]));
const isHuman = (b: string) => b !== "free";

interface Props {
  scenario: string;
  day?: number;
  onSource: (s: Source) => void;
  onOpen: (s: ScreenKey, date?: string) => void;
  onOpenVeteran: (f: VeteranFocus) => void;
}

export function Dashboard({ scenario, day, onSource, onOpen, onOpenVeteran }: Props) {
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [week, setWeek] = useState<(ActionsResponse | null)[]>([]);
  const [headroom, setHeadroom] = useState<(ActionsResponse | null)[]>([]);
  const [sel, setSel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<Source>("fixture");

  const load = useCallback(async () => {
    setError(null);
    try {
      const fc = await getForecast(scenario, day);
      setForecast(fc);
      const dates = fc.dates.slice(0, 7);
      // Open on the event, not on the quiet morning in front of it: the dashboard is
      // about the event, and an empty high-risk tier is the wrong first impression. The
      // strip still starts at today and says which day is selected.
      const ds = eventDays(dates, fc.zips);
      const first = ds.findIndex((d) => d.flood || d.heat || d.smoke);
      setSel(first >= 0 ? first : 0);
      setWeek(new Array(dates.length).fill(null));
      setHeadroom(new Array(dates.length).fill(null));
      // The selected day first, so the screen fills where the eye is.
      const openIdx = first >= 0 ? first : 0;
      const lead = await postActions({ date: dates[openIdx], capacity: DEFAULT_CAPACITY, scenario });
      setWeek((w) => w.map((x, i) => (i === openIdx ? { ...lead, date: dates[openIdx] } : x)));
      setSource(lastSource());
      onSource(lastSource());
      const rest = dates.filter((_, i) => i !== openIdx);
      await Promise.all([
        postActionsWeek(rest, { capacity: DEFAULT_CAPACITY, scenario }, (_i, r) => {
          const j = dates.indexOf(r.date);
          setWeek((w) => w.map((x, k) => (k === j ? r : x)));
        }),
        postActionsWeek(dates, { capacity: UNCAPPED, scenario }, (i, r) => setHeadroom((h) => h.map((x, j) => (j === i ? r : x))), 1),
      ]);
      setSource(lastSource());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [scenario, day, onSource]);

  useEffect(() => {
    void load();
  }, [load]);

  const days = useMemo(() => (forecast ? eventDays(forecast.dates.slice(0, 7), forecast.zips) : []), [forecast]);
  const down = useMemo(() => forecast?.facilities.filter((f) => f.site_down) ?? [], [forecast]);
  const ev = useMemo(() => summarise(days, down.length), [days, down.length]);
  const resp = week[sel] ?? week[0] ?? null;
  const cur = days[sel];

  /** The library, filtered to the event in progress. */
  const evKind: EventKind = ev.kind === "none" ? "all" : ev.kind;
  const resources = useMemo(() => searchLibrary("", evKind).slice(0, 4), [evKind]);

  const askCtx: AskContext = useMemo(() => ({
    days: days.map((d) => ({ date: d.date, flood: d.flood, heat: d.heat, smoke: d.smoke, surge: d.surge, outage: d.outage, maxHeatIndex: d.maxHeatIndex, maxPm25: d.maxPm25 })),
    week,
    facilities: forecast?.facilities ?? [],
    panel: week[0]?.n_panel ?? 0,
  }), [days, week, forecast]);

  const missed = useMemo(() => {
    const free = headroom[sel];
    if (!resp || !free) return null;
    const reached = new Set(resp.actions.filter((a) => isHuman(a.capacity_bucket)).map((a) => a.veteran_id));
    const want = new Set(free.actions.filter((a) => isHuman(a.capacity_bucket) && (a.tier === "act_now" || a.tier === "find_out")).map((a) => a.veteran_id));
    let n = 0;
    want.forEach((v) => {
      if (!reached.has(v)) n += 1;
    });
    return n;
  }, [resp, headroom, sel]);

  if (error) return <div className="page"><div className="empty"><div className="eh">The dashboard did not load</div>{error}</div></div>;
  if (!forecast || !resp || !cur) {
    return (
      <div className="page dash">
        <div className="sk" style={{ height: 380, borderRadius: 22 }} />
        <div className="row g2"><div className="sk" style={{ height: 460 }} /><div className="sk" style={{ height: 460 }} /></div>
      </div>
    );
  }

  const human = resp.actions.filter((a) => isHuman(a.capacity_bucket)).length;
  const need = cur.heat && !cur.flood ? "heat" : "treatment_gap";

  return (
    <div className="page dash">
      <div className="dashbar">
        <div className="db-t">
          <span className="db-live"><span className="pulse" />{source === "api" ? "Live" : "Offline"}</span>
          <b>{fmtDate(days[0].date)}</b>
          <span className="muted">{fmtInt(resp.n_panel)} veterans on the panel · {fmtInt(down.length)} of 14 VA sites closed</span>
        </div>
        <div className="sp" />
        <div className="db-stats">
          <div className="db-s"><b><CountUp value={resp.counts_by_tier.act_now ?? 0} /></b><span>high risk {sel === 0 ? "today" : fmtDate(cur.date)}</span></div>
          <div className="db-s"><b><CountUp value={resp.counts_by_tier.find_out ?? 0} /></b><span>moderate, find out</span></div>
          <div className="db-s"><b><CountUp value={human} /></b><span>to reach by hand {sel === 0 ? "today" : "that day"}</span></div>
          <div className={`db-s${missed ? " warn" : ""}`}><b>{missed === null ? "…" : <CountUp value={missed} />}</b><span>not reached at capacity</span></div>
        </div>
      </div>

      <EventPanel forecast={forecast} days={days} selected={sel} onSelect={setSel} need={need} />

      <div className="row g2">
        <div className="card">
          <div className="ch">
            <h3 style={{ fontSize: 16 }}>At-risk patients · {sel === 0 ? "today, " : ""}{fmtDate(cur.date)}</h3>
            <span className="sub">stratified by risk · hover for the factors, click for the chart</span>
            <span className="sp" />
            <button className="sm" onClick={() => onOpen("careteam", cur.date)}>Full list with capacity <IconArrow /></button>
          </div>
          <PatientList resp={resp} onOpen={onOpenVeteran} />
        </div>

        <div className="stack">
          <AskPanel ctx={askCtx} onGoto={(s, d) => onOpen(s as ScreenKey, d)} compact />

          <div className="card">
            <div className="ch">
              <h3>Resource library</h3>
              <span className="sub">for this event</span>
              <span className="sp" />
              <button className="sm ghost" onClick={() => onOpen("library")}>All resources <IconArrow /></button>
            </div>
            <div className="lib-mini">
              {resources.map((r) => <ResourceCard key={r.id} r={r} open={false} onToggle={() => onOpen("library")} />)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

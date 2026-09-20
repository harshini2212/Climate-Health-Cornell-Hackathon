/**
 * The command center: the screen the care team opens, and the one that goes on the big
 * screen. One screen answers the four questions they actually have.
 *
 *   1. What is coming, how bad, how long have we got   — the storm band and the timeline
 *   2. Who do I reach first                            — the at-risk list, three tiers
 *   3. What changed since I last looked                — the update feed
 *   4. What do I need to know about it                 — Ask Leeward and the library
 *
 * Everything below the band follows the day selected in the timeline, so one click moves
 * the whole screen to landfall day or to the first day of the heat wave.
 *
 * The list is de-identified on its face: a handle, the action, the owner. Names,
 * conditions, driver phrases and the full rationale open on a hover or a click, because
 * this screen hangs in a shared clinical space.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { AskPanel } from "../components/AskPanel";
import { CountUp } from "../components/CountUp";
import { UpdateFeed, chipsFor, eventDays, summarise, type EventDay } from "../components/EventFeed";
import { IconArrow, IconBuilding } from "../components/Icons";
import { NeedMap } from "../components/NeedMap";
import { PatientList } from "../components/PatientList";
import { getForecast, lastSource, postActions, postActionsWeek, type Source } from "../lib/api";
import type { AskContext } from "../lib/ask";
import { searchLibrary, type EventKind } from "../lib/library";
import { BUCKET_LABEL, NEED_LABEL, dayLabel, dayName, fmtDate, fmtInt } from "../lib/labels";
import { DEFAULT_CAPACITY, type ActionRow, type ActionsResponse, type Capacity, type FacilityStatus, type ForecastResponse } from "../lib/types";
import type { ScreenKey } from "../App";
import { ResourceCard } from "./Library";

/** Capacity high enough that nothing is cut. The gap to the real cut is the honest half. */
const UNCAPPED: Capacity = Object.fromEntries(Object.keys(DEFAULT_CAPACITY).map((k) => [k, 100_000]));

/** Human capacity units, in the order the team talks about them. */
const HUMAN_BUCKETS = ["call", "booking", "ride", "va_fill", "pharmacist_slot", "evac", "partner_slot", "refill"] as const;

const isHuman = (bucket: string) => bucket !== "free";

export interface VeteranFocus {
  veteranId: string;
  actionId: string;
  date: string;
}

interface Props {
  scenario: string;
  /** Scenario-relative start day; undefined lets the API open on the landfall window. */
  day?: number;
  onSource: (s: Source) => void;
  onOpen: (s: ScreenKey, date?: string) => void;
  onOpenVeteran: (f: VeteranFocus) => void;
}

interface DayBoard {
  resp: ActionsResponse;
  human: ActionRow[];
  automated: number;
  binding: { bucket: string; used: number; cap: number; frac: number };
  missed: number | null;
  wanted: number | null;
}

function boardFor(resp: ActionsResponse, free: ActionsResponse | null): DayBoard {
  const human = resp.actions.filter((a) => isHuman(a.capacity_bucket));
  const usedBy = new Map<string, number>();
  for (const a of human) usedBy.set(a.capacity_bucket, (usedBy.get(a.capacity_bucket) ?? 0) + 1);
  let binding = { bucket: "call", used: usedBy.get("call") ?? 0, cap: resp.capacity.call ?? 0, frac: 0 };
  binding.frac = binding.cap ? binding.used / binding.cap : 0;
  for (const [bucket, cap] of Object.entries(resp.capacity)) {
    if (!isHuman(bucket) || cap <= 0) continue;
    const used = usedBy.get(bucket) ?? 0;
    const frac = used / cap;
    if (frac > binding.frac) binding = { bucket, used, cap, frac };
  }
  let missed: number | null = null;
  let wanted: number | null = null;
  if (free) {
    const reached = new Set(human.map((a) => a.veteran_id));
    // "Needed a person" means the Act-now and Find-out tiers: at unlimited capacity the
    // allocator would hand almost everyone some human action, which is not the same claim.
    const want = new Set(free.actions.filter((a) => isHuman(a.capacity_bucket) && (a.tier === "act_now" || a.tier === "find_out")).map((a) => a.veteran_id));
    wanted = want.size;
    missed = 0;
    want.forEach((v) => {
      if (!reached.has(v)) missed! += 1;
    });
  }
  return { resp, human: [...human].sort((a, b) => a.rank - b.rank), automated: resp.actions.length - human.length, binding, missed, wanted };
}

export function CommandCenter({ scenario, day, onSource, onOpen, onOpenVeteran }: Props) {
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
      // Open on the event, not on the quiet morning in front of it: an empty High-risk
      // tier is the wrong first thing to see. The timeline still starts at today.
      const ds = eventDays(dates, fc.zips);
      const first = ds.findIndex((d) => d.flood || d.heat || d.smoke);
      const open = first >= 0 ? first : 0;
      setSel(open);
      setWeek(new Array(dates.length).fill(null));
      setHeadroom(new Array(dates.length).fill(null));
      // The selected day first, so the screen fills where the eye already is.
      const lead = await postActions({ date: dates[open], capacity: DEFAULT_CAPACITY, scenario });
      setWeek((w) => w.map((x, i) => (i === open ? { ...lead, date: dates[open] } : x)));
      setSource(lastSource());
      onSource(lastSource());
      await Promise.all([
        postActionsWeek(dates.filter((_, i) => i !== open), { capacity: DEFAULT_CAPACITY, scenario }, (_i, r) => {
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

  /**
   * `facilities` is one row per site per day, so a site down all week arrives five times.
   * A closure is one fact about one site, and the days it covers are the useful part.
   */
  const downSites = useMemo(() => {
    const byId = new Map<string, { f: FacilityStatus; days: string[] }>();
    for (const f of forecast?.facilities ?? []) {
      if (!f.site_down) continue;
      const seen = byId.get(f.facility_id);
      if (seen) seen.days.push((f as FacilityStatus & { date?: string }).date ?? "");
      else byId.set(f.facility_id, { f, days: [(f as FacilityStatus & { date?: string }).date ?? ""] });
    }
    for (const s of byId.values()) s.days.sort();
    return [...byId.values()];
  }, [forecast]);

  const board = useMemo(() => week.map((r, i) => (r ? boardFor(r, headroom[i] ?? null) : null)), [week, headroom]);
  const loaded = board.filter((b): b is DayBoard => b !== null);
  const story = useMemo(() => summarise(days, downSites.length), [days, downSites.length]);

  const lanes = useMemo(() => {
    if (loaded.length === 0) return [];
    return HUMAN_BUCKETS.map((bucket) => {
      const perDay = loaded[0].resp.capacity[bucket] ?? 0;
      const used = loaded.reduce((s, b) => s + b.human.filter((a) => a.capacity_bucket === bucket).length, 0);
      return { bucket, perDay, used, total: perDay * loaded.length };
    }).filter((l) => l.total > 0).sort((a, b) => b.used / b.total - a.used / a.total);
  }, [loaded]);

  const askCtx: AskContext = useMemo(() => ({
    days: days.map((d) => ({ date: d.date, flood: d.flood, heat: d.heat, smoke: d.smoke, surge: d.surge, outage: d.outagePeak, maxHeatIndex: d.maxHeatIndex, maxPm25: d.maxPm25 })),
    week,
    facilities: forecast?.facilities ?? [],
    panel: week.find((r) => r)?.n_panel ?? 0,
  }), [days, week, forecast]);

  const altSiteBookings = loaded.reduce((s, b) => s + b.resp.actions.filter((a) => a.action === "alt_site_booking").length, 0);
  const missedTotal = loaded.reduce((s, b) => s + (b.missed ?? 0), 0);
  const peakDay = board.reduce<DayBoard | null>((best, b) => (b && (!best || (b.resp.counts_by_tier.act_now ?? 0) > (best.resp.counts_by_tier.act_now ?? 0)) ? b : best), null);
  const actNowPeak = peakDay?.resp.counts_by_tier.act_now ?? 0;
  const panel = board.find((b) => b)?.resp.n_panel ?? 0;

  const evKind: EventKind = story.kind === "none" ? "all" : story.kind;
  const resources = useMemo(() => searchLibrary("", evKind).slice(0, 3), [evKind]);

  if (error) return <div className="page"><div className="empty"><div className="eh">The week did not load</div>{error}</div></div>;
  if (!forecast || !board.some((b) => b)) {
    return (
      <div className="page dash">
        <div className="sk" style={{ height: 380, borderRadius: 22 }} />
        <div className="row g2"><div className="sk" style={{ height: 460 }} /><div className="sk" style={{ height: 460 }} /></div>
      </div>
    );
  }

  const cur = board[sel] ?? board.find((b) => b)!;
  const curDay: EventDay = days[sel] ?? days[0];
  const mapNeed = curDay.heat && !curDay.flood ? "heat" : "treatment_gap";

  return (
    <div className="page dash">
      {/* 1 · what is coming, how bad, how long have we got */}
      <section className={`storm sev-${story.severity}`}>
        <div className="storm-head">
          <div className="storm-text">
            <div className="storm-kicker"><span className="pulse" />{source === "api" ? "Live" : "Offline"} · {fmtDate(days[0].date)} · {fmtInt(panel)} veterans on the panel</div>
            <h1 className="storm-title">{story.title}</h1>
            <p className="storm-sub">{story.sub}{downSites.length ? ` · ${downSites.map(({ f }) => `${f.name} (station ${f.facility_id}) closed`).join(", ")}` : ""}</p>
          </div>
          <div className="storm-count">
            <div className="sc-num">{story.daysOut}</div>
            <div className="sc-lab">{story.label}</div>
          </div>
        </div>

        <div className="storm-tiles">
          <div className="st">
            <div className="st-k">High risk at the peak</div>
            <div className="st-v"><CountUp value={actNowPeak} /></div>
            <div className="st-s">{peakDay ? fmtDate(peakDay.resp.date) : "—"} · narrow interval, high risk</div>
          </div>
          <div className={`st${downSites.length ? " st-crit" : ""}`}>
            <div className="st-k">VA sites closed</div>
            <div className="st-v"><CountUp value={downSites.length} /></div>
            <div className="st-s">{downSites.length ? `station ${downSites.map(({ f }) => f.facility_id).join(", ")} · evac zone ${downSites[0].f.evac_zone}` : "all 14 open"}</div>
          </div>
          <div className="st">
            <div className="st-k">Alternate-site bookings</div>
            <div className="st-v"><CountUp value={altSiteBookings} /></div>
            <div className="st-s">dialysis, infusion, OTP moved this week</div>
          </div>
          <div className="st">
            <div className="st-k">People to reach by hand</div>
            <div className="st-v"><CountUp value={loaded.reduce((s, b) => s + b.human.length, 0)} /></div>
            <div className="st-s">across {loaded.length} of {board.length} days · {fmtInt(loaded.reduce((s, b) => s + b.automated, 0))} texts automated</div>
          </div>
          <div className={`st${missedTotal ? " st-warn" : ""}`}>
            <div className="st-k">Not reached at this capacity</div>
            <div className="st-v"><CountUp value={missedTotal} /></div>
            <div className="st-s">High-risk and Find-out veterans without a person</div>
          </div>
        </div>

        <div className="timeline" role="tablist" aria-label="Days">
          {days.map((d, i) => {
            const b = board[i];
            const load = b ? Math.min(1, b.binding.frac) : 0;
            const cs = chipsFor(d);
            return (
              <button key={d.date} className={`tl${i === sel ? " sel" : ""}${d.flood ? " storm-day" : d.heat ? " heat-day" : ""}`} role="tab" aria-selected={i === sel} onClick={() => setSel(i)} disabled={!b} title={b ? `Show ${fmtDate(d.date)}` : "Loading"}>
                <div className="tl-top"><span className="tl-dow">{i === 0 ? "Today" : dayName(d.date)}</span><span className="tl-date">{dayLabel(d.date)}</span></div>
                <div className="tl-temp">{Math.round(d.meanHeatIndex)}<small>°F</small></div>
                <div className="tl-chips">
                  {cs.length === 0 && <span className="hz clear">CLEAR</span>}
                  {cs.slice(0, 2).map((c) => <span key={c.key} className={`hz ${c.key}`} title={c.title}>{c.label}<small>{c.detail}</small></span>)}
                </div>
                <div className={`prog${load >= 1 ? " over" : load >= 0.8 ? " warn" : ""}`}><i style={{ width: `${load * 100}%` }} /></div>
                <div className="tl-load">
                  {b ? <span><b>{b.binding.used}</b>/{b.binding.cap} {BUCKET_LABEL[b.binding.bucket]}</span> : <span>loading…</span>}
                  {b && b.missed !== null && b.missed > 0 && <span className="tl-miss">{b.missed} missed</span>}
                </div>
              </button>
            );
          })}
        </div>
      </section>

      {/* 2 · who do I reach first, and 3 · what changed */}
      <div className="row g2">
        <div className="card">
          <div className="ch">
            <h3 style={{ fontSize: 16 }}>At-risk patients · {sel === 0 ? "today, " : ""}{fmtDate(cur.resp.date)}</h3>
            <span className="sub">stratified by risk · hover for the factors, click for the chart</span>
            <span className="sp" />
            <button className="sm" onClick={() => onOpen("careteam", cur.resp.date)}>Full list with capacity <IconArrow /></button>
          </div>
          <PatientList resp={cur.resp} onOpen={onOpenVeteran} />
        </div>

        <div className="stack">
          {downSites.map(({ f, days: dd }) => (
            <div key={f.facility_id} className="insight">
              <div className="ic ic-critical"><IconBuilding /></div>
              <div className="bd">
                <div className="t">{f.name} (station {f.facility_id}) is closed this week</div>
                <div className="d">{dd.length > 1 ? `${dd.length} days of this window. ` : ""}Evacuation zone {f.evac_zone}{f.site_dependent_services ? "; dialysis, infusion and the opioid treatment program run on site" : ""}. {altSiteBookings} alternate-site {altSiteBookings === 1 ? "booking is" : "bookings are"} queued.</div>
              </div>
            </div>
          ))}

          <div className="card tight">
            <div className="ch" style={{ padding: "14px 18px 0" }}><h3>Updates</h3><span className="sub">click one to move the screen to that day</span></div>
            <UpdateFeed days={days} forecast={forecast} selected={sel} onSelect={setSel} />
          </div>

          <div className="card tight">
            <div className="minimap"><NeedMap forecast={forecast} date={cur.resp.date} need={mapNeed} compact /></div>
            <div className="minimap-cap">
              <span><b>{NEED_LABEL[mapNeed]}</b> expected per ZIP · {fmtDate(cur.resp.date)}</span>
              <button className="sm ghost" onClick={() => onOpen("map")}>Full map <IconArrow /></button>
            </div>
          </div>

          <div className="card">
            <div className="ch"><h3>Capacity this week</h3><span className="sub">by unit · {loaded.length} of {board.length} days</span></div>
            <div className="lanes">
              {lanes.slice(0, 5).map((l) => {
                const frac = Math.min(1, l.used / l.total);
                return (
                  <div key={l.bucket} className="lane">
                    <span className="lb" style={{ textTransform: "capitalize" }}>{BUCKET_LABEL[l.bucket]}<small>{l.perDay} a day</small></span>
                    <div className={`prog${frac >= 1 ? " over" : frac >= 0.8 ? " warn" : ""}`}><i style={{ width: `${frac * 100}%` }} /></div>
                    <span className="bv"><b>{fmtInt(l.used)}</b><span className="muted"> / {fmtInt(l.total)}</span></span>
                  </div>
                );
              })}
            </div>
          </div>

          {source === "fixture" && (
            <div className="callout warn">Offline fixtures: hazards are per day, but the queue is one modelled day repeated. The live API returns a distinct list per day.</div>
          )}
        </div>
      </div>

      {/* 4 · what do I need to know about it */}
      <div className="row g2" style={{ alignItems: "start" }}>
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
  );
}

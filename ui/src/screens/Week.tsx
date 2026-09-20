/**
 * The week board, the demo's opening screen. A care team glances at it and asks two
 * questions: what does the week look like, and who do I reach first. The ribbon answers
 * the first for every day; the queue answers the second for the day selected.
 *
 * The queue is de-identified: a handle, the action, the owner. Names, conditions, driver
 * phrases and the rationale sentence live on the veteran card, one click away, because
 * this screen hangs in a shared clinical space.
 *
 * Loading order matters on stage: today's queue lands first, then the other six days
 * fill in behind it, two at a time, because the API allocates one day at a time.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { IconAlert, IconArrow, IconBuilding } from "../components/Icons";
import { getForecast, lastSource, postActions, postActionsWeek, type Source } from "../lib/api";
import { scenarioMeta } from "../lib/config";
import { ACTION_LABEL, BUCKET_LABEL, OWNER_LABEL, TIER_BADGE, TIER_LABEL, dayLabel, dayName, fmtDate, fmtInt, handleFor } from "../lib/labels";
import { DEFAULT_CAPACITY, type ActionRow, type ActionsResponse, type Capacity, type ForecastResponse, type ZipHazard } from "../lib/types";

/** Capacity high enough that nothing is cut. The gap to the real cut is the honest half. */
const UNCAPPED: Capacity = Object.fromEntries(Object.keys(DEFAULT_CAPACITY).map((k) => [k, 100_000]));

/** Human capacity units, in the order the team talks about them. */
const HUMAN_BUCKETS = ["call", "booking", "ride", "va_fill", "pharmacist_slot", "evac", "partner_slot", "refill"] as const;

interface DayHazard {
  date: string;
  heat: boolean;
  smoke: boolean;
  flood: boolean;
  surge: number;
  outage: number;
  maxHeatIndex: number;
  maxPm25: number;
  evacZone: number;
}

function summarise(dates: string[], zips: ZipHazard[]): DayHazard[] {
  const byDate = new Map<string, DayHazard>();
  for (const d of dates) byDate.set(d, { date: d, heat: false, smoke: false, flood: false, surge: 0, outage: 0, maxHeatIndex: 0, maxPm25: 0, evacZone: 0 });
  for (const z of zips) {
    const d = byDate.get(z.date);
    if (!d) continue;
    d.heat ||= z.heat_alert;
    d.smoke ||= z.smoke_alert;
    d.flood ||= z.flood_warning || z.flash_flood_emergency;
    d.surge = Math.max(d.surge, z.surge_ft);
    d.outage = Math.max(d.outage, z.outage_frac);
    d.maxHeatIndex = Math.max(d.maxHeatIndex, z.heat_index_max_f);
    d.maxPm25 = Math.max(d.maxPm25, z.pm25);
    d.evacZone = Math.max(d.evacZone, z.evac_zone_ordered);
  }
  return dates.map((d) => byDate.get(d)!);
}

/** Words, not pictograms: a two-metre read beats an icon nobody has learned yet. */
function chips(d: DayHazard) {
  const out: { key: string; label: string; detail: string }[] = [];
  if (d.flood) out.push({ key: "flood", label: "FLOOD", detail: d.evacZone ? `evac zone ${d.evacZone}` : "warning" });
  if (d.surge > 0) out.push({ key: "surge", label: "SURGE", detail: `${d.surge.toFixed(0)} ft` });
  if (d.heat) out.push({ key: "heat", label: "HEAT", detail: `${Math.round(d.maxHeatIndex)}°F index` });
  if (d.smoke) out.push({ key: "smoke", label: "SMOKE", detail: `PM2.5 ${Math.round(d.maxPm25)}` });
  if (d.outage > 0.05) out.push({ key: "outage", label: "OUTAGE", detail: `${Math.round(d.outage * 100)}% of ZIPs` });
  return out.slice(0, 3);
}

const isHuman = (bucket: string) => bucket !== "free";

export interface VeteranFocus {
  veteranId: string;
  actionId: string;
  date: string;
}

interface Props {
  scenario: string;
  onSource: (s: Source) => void;
  onOpenDay: (date: string) => void;
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

export function Week({ scenario, onSource, onOpenDay, onOpenVeteran }: Props) {
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [week, setWeek] = useState<(ActionsResponse | null)[]>([]);
  const [headroom, setHeadroom] = useState<(ActionsResponse | null)[]>([]);
  const [sel, setSel] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<Source>("fixture");

  const load = useCallback(async () => {
    setError(null);
    try {
      const fc = await getForecast(scenario, scenarioMeta(scenario).day);
      setForecast(fc);
      const dates = fc.dates.slice(0, 7);
      setWeek(new Array(dates.length).fill(null));
      setHeadroom(new Array(dates.length).fill(null));
      // Today first, so the queue is on screen in well under a second.
      const today = await postActions({ date: dates[0], capacity: DEFAULT_CAPACITY, scenario });
      setWeek((w) => w.map((x, i) => (i === 0 ? { ...today, date: dates[0] } : x)));
      setSource(lastSource());
      onSource(lastSource());
      const rest = dates.slice(1);
      await Promise.all([
        postActionsWeek(rest, { capacity: DEFAULT_CAPACITY, scenario }, (i, r) => setWeek((w) => w.map((x, j) => (j === i + 1 ? r : x)))),
        postActionsWeek(dates, { capacity: UNCAPPED, scenario }, (i, r) => setHeadroom((h) => h.map((x, j) => (j === i ? r : x))), 1),
      ]);
      setSource(lastSource());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [scenario, onSource]);

  useEffect(() => {
    void load();
  }, [load]);

  const days = useMemo(() => (forecast ? summarise(forecast.dates.slice(0, 7), forecast.zips) : []), [forecast]);
  const downSites = useMemo(() => (forecast?.facilities ?? []).filter((f) => f.site_down), [forecast]);
  const board = useMemo(() => week.map((r, i) => (r ? boardFor(r, headroom[i] ?? null) : null)), [week, headroom]);
  const loaded = board.filter((b): b is DayBoard => b !== null);

  /** Capacity by unit over the loaded days: what ran out, and where there was room. */
  const lanes = useMemo(() => {
    if (loaded.length === 0) return [];
    return HUMAN_BUCKETS.map((bucket) => {
      const perDay = loaded[0].resp.capacity[bucket] ?? 0;
      const used = loaded.reduce((s, b) => s + b.human.filter((a) => a.capacity_bucket === bucket).length, 0);
      return { bucket, perDay, used, total: perDay * loaded.length };
    }).filter((l) => l.total > 0).sort((a, b) => b.used / b.total - a.used / a.total);
  }, [loaded]);

  const altSiteBookings = loaded.reduce((s, b) => s + b.resp.actions.filter((a) => a.action === "alt_site_booking").length, 0);
  const missedKnown = loaded.filter((b) => b.missed !== null);
  const missedTotal = missedKnown.reduce((s, b) => s + (b.missed ?? 0), 0);
  const missedMax = Math.max(1, ...missedKnown.map((b) => b.missed ?? 0));

  if (error) return <div className="page"><div className="empty"><div className="eh">The week did not load</div>{error}</div></div>;
  if (!forecast || !board[0]) {
    return (
      <div className="page dash">
        <div className="skgrid" style={{ gridTemplateColumns: "repeat(7, minmax(0,1fr))" }}>{Array.from({ length: 7 }, (_, i) => <div key={i} className="sk" style={{ height: 118 }} />)}</div>
        <div className="row g2"><div className="sk" style={{ height: 420 }} /><div className="sk" style={{ height: 420 }} /></div>
      </div>
    );
  }

  const day = board[sel] ?? board[0];
  const groups = (["care_team", "pharmacist", "partner"] as const).map((k) => ({ key: k, rows: day.human.filter((a) => a.owner === k) })).filter((g) => g.rows.length > 0);

  return (
    <div className="page dash">
      <div className="ribbon" role="tablist" aria-label="Days">
        {days.map((d, i) => {
          const b = board[i];
          const load = b ? Math.min(1, b.binding.frac) : 0;
          return (
            <button key={d.date} className={`rday${i === sel ? " sel" : ""}`} role="tab" aria-selected={i === sel} onClick={() => setSel(i)} disabled={!b} title={b ? `Show the queue for ${fmtDate(d.date)}` : "Loading"}>
              <div className="rd-top">
                <span className="rd-dow">{i === 0 ? "Today" : dayName(d.date)}</span>
                <span className="rd-date">{dayLabel(d.date)}</span>
              </div>
              <div className="rd-chips">
                {chips(d).length === 0 && <span className="hz clear">CLEAR</span>}
                {chips(d).map((c) => (
                  <span key={c.key} className={`hz ${c.key}`}>{c.label}<small>{c.detail}</small></span>
                ))}
              </div>
              <div className={`prog${load >= 1 ? " over" : load >= 0.8 ? " warn" : ""}`}><i style={{ width: `${load * 100}%` }} /></div>
              <div className="rd-load">
                {b ? <span><b>{b.binding.used}</b> of {b.binding.cap} {BUCKET_LABEL[b.binding.bucket]}</span> : <span>loading…</span>}
                {b && b.missed !== null && b.missed > 0 && <span style={{ color: "var(--crit)", fontWeight: 600 }}>{b.missed} missed</span>}
              </div>
            </button>
          );
        })}
      </div>

      <div className="row g2">
        <div className="card">
          <div className="ch">
            <h3>Queue · {sel === 0 ? "today, " : ""}{fmtDate(day.resp.date)}</h3>
            <span className="sub">{day.human.length} people to reach, in priority order</span>
            <span className="sp" />
            <button className="sm" onClick={() => onOpenDay(day.resp.date)}>Full list with capacity <IconArrow /></button>
          </div>
          <div className="qlist">
            {groups.map((g) => (
              <div key={g.key}>
                <div className="qgroup">{OWNER_LABEL[g.key]}<span className="n">{g.rows.length}</span></div>
                {g.rows.map((a: ActionRow, i: number) => (
                  <button key={a.action_id} className="qrow" onClick={() => onOpenVeteran({ veteranId: a.veteran_id, actionId: a.action_id, date: day.resp.date })} title="Open the veteran card">
                    <span className="rk">{i + 1}</span>
                    <span className="hd">{handleFor(a.name_display, a.veteran_id)}</span>
                    <span className="ac"><b>{ACTION_LABEL[a.action] ?? a.action}</b><span>{a.borough} · {BUCKET_LABEL[a.capacity_bucket]?.replace(/s$/, "")} slot</span></span>
                    <span className={TIER_BADGE[a.tier]}>{TIER_LABEL[a.tier]}</span>
                    <IconArrow />
                  </button>
                ))}
              </div>
            ))}
            {day.automated > 0 && <div className="qmore">+ {fmtInt(day.automated)} verified texts go out automatically, no human cost</div>}
            {day.human.length === 0 && <div className="empty sm"><div className="eh">Nothing to reach by hand</div>Only automated texts on this day.</div>}
          </div>
        </div>

        <div className="stack">
          {downSites.map((f) => (
            <div key={f.facility_id} className="insight">
              <div className="ic ic-critical"><IconBuilding /></div>
              <div className="bd">
                <div className="t">{f.name} (station {f.facility_id}) is closed this week</div>
                <div className="d">Evacuation zone {f.evac_zone}{f.site_dependent_services ? "; dialysis, infusion and the opioid treatment program run on site" : ""}. {altSiteBookings} alternate-site {altSiteBookings === 1 ? "booking is" : "bookings are"} queued{loaded.length < board.length ? " so far" : ""}.</div>
              </div>
            </div>
          ))}
          {forecast.headline && (
            <div className="insight">
              <div className="ic ic-medium"><IconAlert /></div>
              <div className="bd"><div className="t">Forecast</div><div className="d">{forecast.headline}</div></div>
            </div>
          )}

          <div className="card">
            <div className="ch"><h3>Capacity this week</h3><span className="sub">by unit · {loaded.length} of {board.length} days loaded</span></div>
            <div className="lanes">
              {lanes.map((l) => {
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

          <div className="card">
            <div className="ch"><h3>Not reached at this capacity</h3><span className="sub">per day</span><span className="sp" /><span className={`pillbadge ${missedTotal > 0 ? "pill-critical" : "pill-healthy"}`}>{fmtInt(missedTotal)} this week</span></div>
            <div className="minibars">
              {board.map((b, i) => (
                <div key={i} className="mb" title={b && b.missed !== null ? `${fmtDate(b.resp.date)}: ${b.missed} of ${b.wanted} who needed a person` : "loading"}>
                  <b>{b && b.missed !== null ? b.missed : "·"}</b>
                  <i className={b && b.missed ? "miss" : ""} style={{ height: `${b && b.missed !== null ? Math.max(3, (100 * b.missed) / missedMax) : 3}%` }} />
                  <span>{i === 0 ? "Today" : days[i] ? dayName(days[i].date) : ""}</span>
                </div>
              ))}
            </div>
            <div className="muted small" style={{ marginTop: 10 }}>Veterans the allocator wanted a person to reach, minus those the day's capacity allowed.</div>
          </div>

          {source === "fixture" && (
            <div className="callout warn">Offline fixtures: hazards are per day, but the queue is one modelled day repeated. The live API returns a distinct list per day.</div>
          )}
        </div>
      </div>
    </div>
  );
}

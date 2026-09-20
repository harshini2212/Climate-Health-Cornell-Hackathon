/**
 * The week board, the demo's opening screen and the one on the big screen.
 *
 * The storm band at the top says the one thing the room needs to know, with a countdown
 * to it, the numbers that matter this week counting up, and the seven days as a hazard
 * timeline. Below it: the queue for the selected day, de-identified, and a live map of
 * expected need that re-colours as the day changes.
 *
 * The queue shows a handle, the action, the owner and the card-safe headline. Names,
 * conditions, driver phrases and the full rationale live on the veteran card, one click
 * away, because this screen hangs in a shared clinical space.
 *
 * Loading order matters on stage: today's queue lands first, then the other six days
 * fill in behind it, two at a time, because the API allocates one day at a time.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { CountUp } from "../components/CountUp";
import { IconAlert, IconArrow, IconBuilding } from "../components/Icons";
import { NeedMap } from "../components/NeedMap";
import { getForecast, lastSource, postActions, postActionsWeek, type Source } from "../lib/api";
import { ACTION_LABEL, BUCKET_LABEL, NEED_LABEL, OWNER_LABEL, TIER_BADGE, TIER_LABEL, dayLabel, dayName, fmtDate, fmtInt, handleFor } from "../lib/labels";
import { DEFAULT_CAPACITY, type ActionRow, type ActionsResponse, type Capacity, type FacilityStatus, type ForecastResponse, type ZipHazard } from "../lib/types";

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
  /** ZIPs at or over the outage line, and how many ZIPs the day had at all. */
  outageZips: number;
  nZips: number;
  /** Worst single ZIP's outage fraction. Not a share of ZIPs -- see `chips`. */
  outagePeak: number;
  maxHeatIndex: number;
  meanHeatIndex: number;
  maxPm25: number;
  evacZone: number;
}

/**
 * The line Forecast.tsx already uses to call a ZIP "in outage" (`outage_frac >= 0.2`).
 * Both screens summarise the same field, so they have to summarise it the same way.
 */
const OUTAGE_LINE = 0.2;

function summarise(dates: string[], zips: ZipHazard[]): DayHazard[] {
  const byDate = new Map<string, DayHazard & { sum: number }>();
  for (const d of dates) byDate.set(d, { date: d, heat: false, smoke: false, flood: false, surge: 0, outageZips: 0, nZips: 0, outagePeak: 0, maxHeatIndex: 0, meanHeatIndex: 0, maxPm25: 0, evacZone: 0, sum: 0 });
  for (const z of zips) {
    const d = byDate.get(z.date);
    if (!d) continue;
    d.heat ||= z.heat_alert;
    d.smoke ||= z.smoke_alert;
    d.flood ||= z.flood_warning || z.flash_flood_emergency;
    d.surge = Math.max(d.surge, z.surge_ft);
    d.nZips += 1;
    if (z.outage_frac >= OUTAGE_LINE) d.outageZips += 1;
    d.outagePeak = Math.max(d.outagePeak, z.outage_frac);
    d.maxHeatIndex = Math.max(d.maxHeatIndex, z.heat_index_max_f);
    d.maxPm25 = Math.max(d.maxPm25, z.pm25);
    d.evacZone = Math.max(d.evacZone, z.evac_zone_ordered);
    d.sum += z.heat_index_max_f;
  }
  return dates.map((d) => {
    const x = byDate.get(d)!;
    return { ...x, meanHeatIndex: x.nZips ? x.sum / x.nZips : 0 };
  });
}

/**
 * Words, not pictograms: a two-metre read beats an icon nobody has learned yet.
 *
 * The outage chip used to read "<peak fraction>% of ZIPs", which is two different numbers
 * wearing one label: 0.80 is how dark the worst single ZIP is, not how much of the city is
 * out. On landfall day that printed "80% of ZIPs" when 62% of ZIPs were affected. The
 * share is the number the label promises, so the share is what it now shows; the peak is
 * still worth knowing for powered-equipment patients, so it moves into the hover.
 */
function chips(d: DayHazard) {
  const out: { key: string; label: string; detail: string; title?: string }[] = [];
  if (d.flood) out.push({ key: "flood", label: "FLOOD", detail: d.evacZone ? `evac zone ${d.evacZone}` : "warning" });
  if (d.surge > 0) out.push({ key: "surge", label: "SURGE", detail: `${d.surge.toFixed(0)} ft` });
  if (d.heat) out.push({ key: "heat", label: "HEAT", detail: `${Math.round(d.maxHeatIndex)}°F index` });
  if (d.smoke) out.push({ key: "smoke", label: "SMOKE", detail: `PM2.5 ${Math.round(d.maxPm25)}` });
  if (d.outageZips > 0) {
    const share = d.nZips ? Math.round((100 * d.outageZips) / d.nZips) : 0;
    out.push({
      key: "outage",
      label: "OUTAGE",
      detail: `${share}% of ZIPs`,
      title: `${d.outageZips} of ${d.nZips} ZIPs at or over ${Math.round(OUTAGE_LINE * 100)}% of customers out; worst ZIP ${Math.round(d.outagePeak * 100)}%`,
    });
  }
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
  /** Scenario-relative start day; undefined lets the API open on the landfall window. */
  day?: number;
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

/** The one line the room needs, and how far away it is. */
function storyOf(days: DayHazard[]) {
  const flood = days.findIndex((d) => d.flood);
  const heat = days.findIndex((d) => d.heat);
  const smoke = days.findIndex((d) => d.smoke);
  const FULL: Record<string, string> = { Sun: "Sunday", Mon: "Monday", Tue: "Tuesday", Wed: "Wednesday", Thu: "Thursday", Fri: "Friday", Sat: "Saturday" };
  const when = (i: number) => (i === 0 ? "today" : i === 1 ? "tomorrow" : `on ${FULL[dayName(days[i].date)] ?? dayName(days[i].date)}`);
  if (flood >= 0) {
    return {
      title: `Coastal storm makes landfall ${when(flood)}`,
      sub: heat > flood ? `then a heat wave from ${dayName(days[heat].date)} ${dayLabel(days[heat].date)}, while power is still out` : "surge into evacuation zones 1–2, outages along the waterfront",
      count: flood, label: flood === 0 ? "landfall today" : flood === 1 ? "day to landfall" : "days to landfall",
    };
  }
  if (heat >= 0) return { title: `Heat wave from ${when(heat)}`, sub: `heat index ${Math.round(days[heat].maxHeatIndex)}°F at the peak`, count: heat, label: heat === 1 ? "day to the heat wave" : "days to the heat wave" };
  if (smoke >= 0) return { title: `Wildfire smoke from ${when(smoke)}`, sub: `PM2.5 up to ${Math.round(days[smoke].maxPm25)} µg/m³`, count: smoke, label: "days to the smoke" };
  return { title: "A quiet week", sub: "no alerts in the seven-day window", count: 0, label: "alerts" };
}

export function Week({ scenario, day, onSource, onOpenDay, onOpenVeteran }: Props) {
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
      setSel(0);
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
  }, [scenario, day, onSource]);

  useEffect(() => {
    void load();
  }, [load]);

  const days = useMemo(() => (forecast ? summarise(forecast.dates.slice(0, 7), forecast.zips) : []), [forecast]);
  /**
   * `facilities` is one row per site *per day*, so a site that is down all week arrives
   * five times. Rendering the list raw printed the same closure card five times over and
   * keyed them all on `facility_id`, which is one duplicate React key per repeat. A
   * closure is one fact about one site, and the days it covers are the useful part of it.
   */
  const downSites = useMemo(() => {
    const byId = new Map<string, { f: FacilityStatus; days: string[] }>();
    for (const f of forecast?.facilities ?? []) {
      if (!f.site_down) continue;
      const seen = byId.get(f.facility_id);
      if (seen) seen.days.push(f.date);
      else byId.set(f.facility_id, { f, days: [f.date] });
    }
    for (const s of byId.values()) s.days.sort();
    return [...byId.values()];
  }, [forecast]);
  const board = useMemo(() => week.map((r, i) => (r ? boardFor(r, headroom[i] ?? null) : null)), [week, headroom]);
  const loaded = board.filter((b): b is DayBoard => b !== null);
  const story = useMemo(() => storyOf(days), [days]);

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
  const peakDay = board.reduce<DayBoard | null>((best, b) => (b && (!best || (b.resp.counts_by_tier.act_now ?? 0) > (best.resp.counts_by_tier.act_now ?? 0)) ? b : best), null);
  const actNowPeak = peakDay?.resp.counts_by_tier.act_now ?? 0;
  const panel = board[0]?.resp.n_panel ?? 0;

  if (error) return <div className="page"><div className="empty"><div className="eh">The week did not load</div>{error}</div></div>;
  if (!forecast || !board[0]) {
    return (
      <div className="page dash">
        <div className="sk" style={{ height: 330, borderRadius: 22 }} />
        <div className="row g2"><div className="sk" style={{ height: 420 }} /><div className="sk" style={{ height: 420 }} /></div>
      </div>
    );
  }

  const cur = board[sel] ?? board[0];
  const curDay = days[sel] ?? days[0];
  const groups = (["care_team", "pharmacist", "partner"] as const).map((k) => ({ key: k, rows: cur.human.filter((a) => a.owner === k) })).filter((g) => g.rows.length > 0);
  const mapNeed = curDay.heat && !curDay.flood ? "heat" : "treatment_gap";

  return (
    <div className="page dash">
      <section className="storm">
        <div className="storm-head">
          <div className="storm-text">
            <div className="storm-kicker"><span className="pulse" />{source === "api" ? "Live" : "Offline"} · {fmtDate(days[0].date)} · {fmtInt(panel)} veterans on the panel</div>
            <h1 className="storm-title">{story.title}</h1>
            <p className="storm-sub">{story.sub}{downSites.length ? ` · ${downSites.map(({ f }) => `${f.name} (station ${f.facility_id}) closed`).join(", ")}` : ""}</p>
          </div>
          <div className="storm-count">
            <div className="sc-num">{story.count}</div>
            <div className="sc-lab">{story.label}</div>
          </div>
        </div>

        <div className="storm-tiles">
          <div className="st">
            <div className="st-k">Act now at the peak</div>
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
            <div className="st-s">across {loaded.length} of {board.length} days · {fmtInt(loaded.reduce((s, b) => s + b.automated, 0))} verified texts automated</div>
          </div>
          <div className={`st${missedTotal ? " st-warn" : ""}`}>
            <div className="st-k">Not reached at this capacity</div>
            <div className="st-v"><CountUp value={missedTotal} /></div>
            <div className="st-s">Act-now and Find-out veterans without a person</div>
          </div>
        </div>

        <div className="timeline" role="tablist" aria-label="Days">
          {days.map((d, i) => {
            const b = board[i];
            const load = b ? Math.min(1, b.binding.frac) : 0;
            const cs = chips(d);
            return (
              <button key={d.date} className={`tl${i === sel ? " sel" : ""}${d.flood ? " storm-day" : d.heat ? " heat-day" : ""}`} role="tab" aria-selected={i === sel} onClick={() => setSel(i)} disabled={!b} title={b ? `Show the queue for ${fmtDate(d.date)}` : "Loading"}>
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

      <div className="row g2">
        <div className="card">
          <div className="ch">
            <h3 style={{ fontSize: 16 }}>Queue · {sel === 0 ? "today, " : ""}{fmtDate(cur.resp.date)}</h3>
            <span className="sub">{cur.human.length} people to reach, in priority order</span>
            <span className="sp" />
            <button className="sm" onClick={() => onOpenDay(cur.resp.date)}>Full list with capacity <IconArrow /></button>
          </div>
          <div className="qlist big">
            {groups.map((g) => (
              <div key={g.key}>
                <div className="qgroup">{OWNER_LABEL[g.key]}<span className="n">{g.rows.length}</span></div>
                {g.rows.map((a: ActionRow, i: number) => (
                  <button key={a.action_id} className="qrow" onClick={() => onOpenVeteran({ veteranId: a.veteran_id, actionId: a.action_id, date: cur.resp.date })} title="Open the veteran card">
                    <span className="rk">{i + 1}</span>
                    <span className="hd">{handleFor(a.name_display, a.veteran_id)}</span>
                    <span className="ac"><b>{ACTION_LABEL[a.action] ?? a.action}</b><span>{a.headline}</span></span>
                    <span className={TIER_BADGE[a.tier]}>{TIER_LABEL[a.tier]}</span>
                    <IconArrow />
                  </button>
                ))}
              </div>
            ))}
            {cur.automated > 0 && <div className="qmore">+ {fmtInt(cur.automated)} verified texts go out automatically, no human cost</div>}
            {cur.human.length === 0 && <div className="empty sm"><div className="eh">Nothing to reach by hand</div>Only automated texts on this day.</div>}
          </div>
        </div>

        <div className="stack">
          <div className="card tight">
            <div className="minimap"><NeedMap forecast={forecast} date={cur.resp.date} need={mapNeed} compact /></div>
            <div className="minimap-cap"><span><b>{NEED_LABEL[mapNeed]}</b> expected per ZIP · {fmtDate(cur.resp.date)}</span>{downSites.length > 0 && <span className="rb rb-critical"><IconBuilding /> {downSites.length} site down</span>}</div>
          </div>
          {downSites.map(({ f, days }) => (
            <div key={f.facility_id} className="insight">
              <div className="ic ic-critical"><IconBuilding /></div>
              <div className="bd">
                <div className="t">
                  {f.name} (station {f.facility_id}) is closed {days.length === 1 ? fmtDate(days[0]) : `${fmtDate(days[0])} – ${fmtDate(days[days.length - 1])}`}
                </div>
                <div className="d">{days.length} {days.length === 1 ? "day" : "days"} of this window. Evacuation zone {f.evac_zone}{f.site_dependent_services ? "; dialysis, infusion and the opioid treatment program run on site" : ""}. {altSiteBookings} alternate-site {altSiteBookings === 1 ? "booking is" : "bookings are"} queued{loaded.length < board.length ? " so far" : ""}.</div>
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
            <div className="ch"><h3>Not reached at this capacity</h3><span className="sub">per day</span><span className="sp" /><span className={`pillbadge ${missedTotal > 0 ? "pill-critical" : "pill-healthy"}`}>{fmtInt(missedTotal)} this week</span></div>
            <div className="minibars">
              {board.map((b, i) => (
                <div key={i} className={`mb${i === sel ? " sel" : ""}`} title={b && b.missed !== null ? `${fmtDate(b.resp.date)}: ${b.missed} of ${b.wanted} who needed a person` : "loading"}>
                  <b>{b && b.missed !== null ? b.missed : "·"}</b>
                  <i className={b && b.missed ? "miss" : ""} style={{ height: `${b && b.missed !== null ? Math.max(3, (100 * b.missed) / missedMax) : 3}%` }} />
                  <span>{i === 0 ? "Today" : days[i] ? dayName(days[i].date) : ""}</span>
                </div>
              ))}
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
    </div>
  );
}

/**
 * The week board. The care team does not visit a page, they glance at a side screen and
 * ask two questions: what does next week look like, and who do I reach first. Time is the
 * primary axis; risk only orders the work inside a day.
 *
 * Readable from two metres: nothing lives in a hover, colour carries urgency and never
 * decoration. The queue is de-identified by default because this hangs in a shared
 * clinical space -- names, conditions, driver phrases and medicines open on a click.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { IconAlert, IconArrow } from "../components/Icons";
import { getForecast, lastSource, postActionsWeek, type Source } from "../lib/api";
import { ACTION_LABEL, TIER_BADGE, TIER_LABEL } from "../lib/labels";
import {
  DEFAULT_CAPACITY,
  type ActionsResponse,
  type Capacity,
  type ForecastResponse,
  type ZipHazard,
} from "../lib/types";

/** What a capacity unit is called out loud. The ribbon names the one that ran out. */
const BUCKET_LABEL: Record<string, string> = {
  call: "calls", refill: "refills", ride: "rides", booking: "bookings", evac: "evac slots",
  partner_slot: "partner slots", pharmacist_slot: "pharmacist slots", va_fill: "VA fills",
};

const OWNER_LANES = [
  { key: "care_team", label: "Care team", bucket: "call", note: "calls a day" },
  { key: "pharmacist", label: "Pharmacist", bucket: "pharmacist_slot", note: "a day — a real person's afternoon" },
  { key: "partner", label: "Partner", bucket: "partner_slot", note: "a day" },
  { key: "automated", label: "Automated", bucket: "free", note: "no human cost" },
] as const;

/** Capacity high enough that nothing is cut. The gap to the real cut is the honest half. */
const UNCAPPED: Capacity = Object.fromEntries(
  Object.keys(DEFAULT_CAPACITY).map((k) => [k, 100_000]),
);

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

/** Collapse ~178 ZIP rows for one day into the handful of facts a glyph row can carry. */
function summarise(dates: string[], zips: ZipHazard[]): DayHazard[] {
  const byDate = new Map<string, DayHazard>();
  for (const d of dates) {
    byDate.set(d, {
      date: d, heat: false, smoke: false, flood: false,
      surge: 0, outage: 0, maxHeatIndex: 0, maxPm25: 0, evacZone: 0,
    });
  }
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
function glyphs(d: DayHazard) {
  const out: { key: string; label: string; detail: string; cls: string }[] = [];
  if (d.heat) out.push({ key: "heat", label: "HEAT", detail: `${Math.round(d.maxHeatIndex)}°F index`, cls: "rb-critical" });
  if (d.smoke) out.push({ key: "smoke", label: "SMOKE", detail: `PM2.5 ${Math.round(d.maxPm25)}`, cls: "rb-high" });
  if (d.flood) out.push({ key: "flood", label: "FLOOD", detail: d.evacZone ? `evac zone ${d.evacZone}` : "warning", cls: "rb-critical" });
  if (d.surge > 0) out.push({ key: "surge", label: "SURGE", detail: `${d.surge.toFixed(1)} ft`, cls: "rb-high" });
  if (d.outage > 0.05) out.push({ key: "outage", label: "OUTAGE", detail: `${Math.round(d.outage * 100)}% of ZIPs`, cls: "rb-medium" });
  return out;
}

const DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** Dates are plain ISO strings; parse as local so a column never slips a day. */
function parseDay(iso: string): Date {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, (m ?? 1) - 1, d ?? 1);
}
const dayName = (iso: string) => DOW[parseDay(iso).getDay()];
const dayLabel = (iso: string) => {
  const d = parseDay(iso);
  return `${d.getDate()} ${MON[d.getMonth()]}`;
};

/**
 * "W.O. · 4471". A board on a wall in a shared clinical space shows a handle, never a
 * name. The full card is one click away for the person who needs it.
 */
export function handleFor(nameDisplay: string, veteranId: string): string {
  const initials = nameDisplay
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase())
    .join(".");
  const digits = veteranId.replace(/\D/g, "").slice(-4) || veteranId.slice(-4);
  return `${initials ? `${initials}.` : "—"} · ${digits}`;
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
  /** Clicking a day hands the date to the action list and its capacity slider. */
  onOpenDay: (date: string) => void;
  /** Clicking a card opens the full veteran card. */
  onOpenVeteran: (f: VeteranFocus) => void;
}

export function Week({ scenario, onSource, onOpenDay, onOpenVeteran }: Props) {
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [week, setWeek] = useState<ActionsResponse[] | null>(null);
  const [headroom, setHeadroom] = useState<ActionsResponse[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<Source>("fixture");

  const load = useCallback(async () => {
    setError(null);
    try {
      const fc = await getForecast(scenario, 0);
      setForecast(fc);
      const dates = fc.dates.slice(0, 7);
      // Two passes: the real cut, and the same week with capacity removed. The difference
      // is exactly "who we could not reach at this capacity", which is the honest half of
      // the story and cannot be read off a single response.
      const [real, free] = await Promise.all([
        postActionsWeek(dates, { capacity: DEFAULT_CAPACITY, scenario }),
        postActionsWeek(dates, { capacity: UNCAPPED, scenario }),
      ]);
      setWeek(real);
      setHeadroom(free);
      setSource(lastSource());
      onSource(lastSource());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [scenario, onSource]);

  useEffect(() => {
    void load();
  }, [load]);

  const days = useMemo(
    () => (forecast ? summarise(forecast.dates.slice(0, 7), forecast.zips) : []),
    [forecast],
  );

  const downSites = useMemo(
    () => (forecast?.facilities ?? []).filter((f) => f.site_down),
    [forecast],
  );

  /** Per day: the queue, the bucket the day is bound by, and who did not fit. */
  const board = useMemo(() => {
    if (!week) return [];
    return week.map((resp, i) => {
      const human = resp.actions.filter((a) => isHuman(a.capacity_bucket));
      // The day is as heavy as its fullest bucket. Summing every bucket would read
      // half-empty on a day when the calls ran out and only the refills had room.
      const usedBy = new Map<string, number>();
      for (const a of human) usedBy.set(a.capacity_bucket, (usedBy.get(a.capacity_bucket) ?? 0) + 1);
      let binding = { bucket: "call", used: 0, cap: resp.capacity.call ?? 0, frac: 0 };
      for (const [bucket, cap] of Object.entries(resp.capacity)) {
        if (!isHuman(bucket) || cap <= 0) continue;
        const used = usedBy.get(bucket) ?? 0;
        const frac = used / cap;
        if (frac > binding.frac) binding = { bucket, used, cap, frac };
      }
      const reached = new Set(human.map((a) => a.veteran_id));
      const wanted = headroom?.[i]
        ? new Set(headroom[i].actions.filter((a) => isHuman(a.capacity_bucket)).map((a) => a.veteran_id))
        : new Set<string>();
      let missed = 0;
      wanted.forEach((v) => {
        if (!reached.has(v)) missed += 1;
      });
      return { resp, actions: [...resp.actions].sort((a, b) => a.rank - b.rank), binding, missed };
    });
  }, [week, headroom]);

  /** Owner lanes are a week total, because the board's axis is the week. */
  const lanes = useMemo(() => {
    if (!week) return [];
    return OWNER_LANES.map((lane) => {
      const used = week.reduce(
        (s, r) => s + r.actions.filter((a) => a.owner === lane.key).length, 0);
      const perDay = week[0]?.capacity[lane.bucket] ?? 0;
      return { ...lane, used, perDay, total: perDay * week.length };
    });
  }, [week]);

  const altSiteBookings = useMemo(
    () => (week ?? []).reduce(
      (s, r) => s + r.actions.filter((a) => a.action === "alt_site_booking").length, 0),
    [week],
  );

  if (error) {
    return (
      <div className="page">
        <div className="empty"><div className="eh">The week did not load</div>{error}</div>
      </div>
    );
  }
  if (!forecast || !week) return <div className="page"><div className="spin">Building the week…</div></div>;

  return (
    <div className="page dash">
      {/* 1 · the ribbon: why Thursday is heavy, answered before anyone asks */}
      <div className="ribbon">
        {days.map((d, i) => {
          const b = board[i];
          const load = b ? Math.min(1, b.binding.frac) : 0;
          return (
            <button
              key={d.date}
              className={`rday${i === 0 ? " today" : ""}`}
              onClick={() => onOpenDay(d.date)}
              title={`Open the action list for ${d.date}`}
            >
              <div className="rd-top">
                <span className="rd-dow">{i === 0 ? "Today" : dayName(d.date)}</span>
                <span className="rd-date">{dayLabel(d.date)}</span>
              </div>
              <div className="rd-glyphs">
                {glyphs(d).length === 0 && <span className="rb rb-low">clear</span>}
                {glyphs(d).map((g) => (
                  <span key={g.key} className={`rb ${g.cls} rd-g`}>
                    {g.label}
                    <small>{g.detail}</small>
                  </span>
                ))}
              </div>
              <div className={`prog${load >= 1 ? " over" : load >= 0.8 ? " warn" : ""}`}>
                <i style={{ width: `${load * 100}%` }} />
              </div>
              <div className="rd-load">
                {b ? `${b.binding.used} of ${b.binding.cap} ${BUCKET_LABEL[b.binding.bucket] ?? b.binding.bucket}` : "—"}
              </div>
            </button>
          );
        })}
      </div>

      {/* 2 · standing banners: a team-wide event is said once, not once per veteran */}
      <div className="banners">
        {forecast.headline && (
          <div className="banner">
            <span className="rb rb-medium">Forecast</span>
            <span>{forecast.headline}</span>
          </div>
        )}
        {downSites.map((f) => (
          <div key={f.facility_id} className="banner crit">
            <span className="rb rb-critical">Site closed</span>
            <span>
              <b>{f.name} ({f.facility_id})</b> is closed in this window
              {f.site_dependent_services && ", and it carries site-dependent care"}.{" "}
              {altSiteBookings} alternate-site{" "}
              {altSiteBookings === 1 ? "booking is" : "bookings are"} queued this week.
            </span>
          </div>
        ))}
        {/* The offline candidate list is one modelled day. The ribbon above is genuinely
            per-day from /forecast, but nobody should read seven forecasts into the queue. */}
        {source === "fixture" && (
          <div className="banner">
            <span className="rb rb-quiet">Fixtures</span>
            <span>
              Hazards and capacity are per-day; the queue is one modelled day ({week[0].date})
              repeated, because the offline candidate list is single-day. The live API returns
              a distinct list per day.
            </span>
          </div>
        )}
      </div>

      {/* 4 · owner lanes: the pharmacist lane is scarce on purpose, so show it */}
      <div className="strip">
        {lanes.map((l) => {
          const frac = l.total > 0 ? Math.min(1, l.used / l.total) : 0;
          const scarce = l.bucket !== "free" && frac >= 0.9;
          return (
            <div key={l.key} className={`stat${scarce ? " crit" : ""}`}>
              <div className="k">{l.label}</div>
              <div className="vrow">
                <span className="v">{l.used}</span>
                {l.bucket !== "free" && (
                  <span className="muted" style={{ fontSize: 12 }}>of {l.total}</span>
                )}
              </div>
              {l.bucket !== "free" && (
                <div className={`prog${scarce ? " over" : ""}`} style={{ marginTop: 9 }}>
                  <i style={{ width: `${frac * 100}%` }} />
                </div>
              )}
              <div className="s">{l.bucket === "free" ? l.note : `${l.perDay} ${l.note}`}</div>
            </div>
          );
        })}
      </div>

      {/* 3 · the queue, under the day it is due, priority order inside the day */}
      <div className="queue">
        {board.map((b, i) => (
          <section key={b.resp.date} className="qcol">
            <header onClick={() => onOpenDay(b.resp.date)}>
              <span>{i === 0 ? "Today" : dayName(b.resp.date)}</span>
              <span className="muted">{b.actions.length}</span>
              <IconArrow />
            </header>
            <div className="qcards">
              {b.actions.map((a) => (
                <button
                  key={a.action_id}
                  className="qcard"
                  onClick={() =>
                    onOpenVeteran({ veteranId: a.veteran_id, actionId: a.action_id, date: b.resp.date })
                  }
                >
                  <div className="qc-top">
                    <span className="qc-handle">{handleFor(a.name_display, a.veteran_id)}</span>
                    <span className={TIER_BADGE[a.tier]}>{TIER_LABEL[a.tier]}</span>
                  </div>
                  <div className="qc-action">{ACTION_LABEL[a.action] ?? a.action}</div>
                  {/* `headline`, never `rationale`: the rich sentence names the service
                      and the driver, and this card is two metres from the corridor. */}
                  <div className="qc-why">{a.headline}</div>
                  <div className="qc-owner">{a.owner.replace(/_/g, " ")}</div>
                </button>
              ))}
              {b.actions.length === 0 && (
                <div className="muted" style={{ fontSize: 12, padding: 8 }}>Nothing queued</div>
              )}
            </div>
            {/* 5 · what does not fit. More persuasive than the list above it. */}
            <footer className={b.missed > 0 ? "short" : ""}>
              {b.missed > 0 ? <IconAlert /> : null}
              {b.missed > 0
                ? `${b.missed} veteran${b.missed === 1 ? "" : "s"} not reached at this capacity`
                : "Everyone who needed a person got one"}
            </footer>
          </section>
        ))}
      </div>
    </div>
  );
}

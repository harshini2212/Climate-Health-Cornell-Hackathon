/**
 * Ask Leeward. Answers are computed from the data on screen and the resource library,
 * deterministically, with no model call and no network: the demo runs with the wifi off
 * and the same question gives the same answer on stage. The panel says so.
 */

import { searchLibrary, type Resource } from "./library";
import { ACTION_LABEL, BUCKET_LABEL, TIER_LABEL, dayName, fmtDate, fmtInt, handleFor } from "./labels";
import type { ActionRow, ActionsResponse, FacilityStatus } from "./types";

export interface DaySummary {
  date: string;
  flood: boolean;
  heat: boolean;
  smoke: boolean;
  surge: number;
  outage: number;
  maxHeatIndex: number;
  maxPm25: number;
}

export interface AskContext {
  days: DaySummary[];
  /** One response per day, null while loading. */
  week: (ActionsResponse | null)[];
  facilities: FacilityStatus[];
  panel: number;
}

export interface Answer {
  text: string;
  bullets?: string[];
  resources?: Resource[];
  /** A screen the answer points at, if any. */
  goto?: { screen: string; label: string; date?: string };
}

export const SUGGESTED = [
  "Who should we call first on landfall day?",
  "Which VA sites are closed and who does that affect?",
  "How many veterans are at high risk this week?",
  "What do diuretics do in a heat wave?",
  "Which patients on oxygen need a backup-power plan?",
  "How do veterans know a message is really from the VA?",
];

const has = (q: string, ...words: string[]) => words.some((w) => q.includes(w));

function dayFor(q: string, ctx: AskContext): number {
  const names: Record<string, number> = { sun: 0, mon: 1, tue: 2, wed: 3, thu: 4, fri: 5, sat: 6 };
  for (const [k, dow] of Object.entries(names)) {
    if (q.includes(k)) {
      const i = ctx.days.findIndex((d) => dayName(d.date).toLowerCase().startsWith(k) && dow >= 0);
      if (i >= 0) return i;
    }
  }
  if (has(q, "landfall", "storm", "flood")) {
    const i = ctx.days.findIndex((d) => d.flood);
    if (i >= 0) return i;
  }
  if (has(q, "heat wave", "heat")) {
    const i = ctx.days.findIndex((d) => d.heat);
    if (i >= 0) return i;
  }
  if (has(q, "tomorrow")) return Math.min(1, ctx.days.length - 1);
  return 0;
}

const isHuman = (a: ActionRow) => a.capacity_bucket !== "free";

export function answer(question: string, ctx: AskContext): Answer {
  const q = question.toLowerCase().trim();
  const i = dayFor(q, ctx);
  const day = ctx.days[i];
  const resp = ctx.week[i];
  const down = ctx.facilities.filter((f) => f.site_down);
  const when = day ? fmtDate(day.date) : "today";

  if (has(q, "call first", "who first", "priority", "top of", "first call", "start with", "who should")) {
    if (!resp) return { text: `The list for ${when} is still loading.` };
    const top = resp.actions.filter(isHuman).slice(0, 5);
    return {
      text: `${when}: ${fmtInt(resp.actions.filter(isHuman).length)} people to reach by hand, ranked by expected harm averted. The first five:`,
      bullets: top.map((a, k) => `${k + 1}. ${a.name_display} (${handleFor(a.name_display, a.veteran_id)}), ${a.borough}: ${ACTION_LABEL[a.action] ?? a.action}, ${TIER_LABEL[a.tier]}. ${a.headline}`),
      goto: { screen: "careteam", label: `Open the full list for ${when}`, date: day?.date },
    };
  }

  if (has(q, "closed", "site down", "sites are", "which site", "630", "dialysis site", "manhattan va")) {
    if (down.length === 0) return { text: "All 14 NYC VA facilities are open in this window." };
    const bookings = ctx.week.reduce((s, r) => s + (r?.actions.filter((a) => a.action === "alt_site_booking").length ?? 0), 0);
    return {
      text: `${down.map((f) => `${f.name} (station ${f.facility_id}, evacuation zone ${f.evac_zone})`).join("; ")} ${down.length === 1 ? "is" : "are"} closed in this window.`,
      bullets: [
        `${down.some((f) => f.site_dependent_services) ? "The site carries site-dependent care: dialysis, infusion and the opioid treatment program. " : ""}${fmtInt(bookings)} alternate-site bookings are queued across the week.`,
        "Controlled-substance patients cannot use the retail emergency refill, so OTP patients rank first.",
      ],
      resources: searchLibrary("site closed dialysis", "flood"),
      goto: { screen: "map", label: "See the sites on the map" },
    };
  }

  if (has(q, "how many", "at risk", "act now", "high risk", "count")) {
    const rows = ctx.week.map((r, k) => (r ? `${k === 0 ? "Today" : dayName(ctx.days[k].date)} ${fmtDate(ctx.days[k].date).slice(4)}: ${r.counts_by_tier.act_now ?? 0} act now, ${r.counts_by_tier.find_out ?? 0} find out, ${fmtInt(r.counts_by_tier.self_serve ?? 0)} self-serve` : null)).filter((x): x is string => x !== null);
    const peak = ctx.week.reduce((b, r) => (r && (r.counts_by_tier.act_now ?? 0) > (b?.counts_by_tier.act_now ?? 0) ? r : b), null as ActionsResponse | null);
    return {
      text: `${fmtInt(ctx.panel)} veterans are on the panel. ${peak ? `The peak is ${fmtDate(peak.date)} with ${peak.counts_by_tier.act_now ?? 0} in the Act-now tier.` : ""}`,
      bullets: rows,
      goto: { screen: "week", label: "Open the week board" },
    };
  }

  if (has(q, "capacity", "calls", "not reached", "missed", "enough")) {
    if (!resp) return { text: "Capacity for that day is still loading." };
    const used = resp.actions.filter((a) => a.capacity_bucket === "call").length;
    const byBucket = Object.entries(resp.capacity).filter(([b]) => b !== "free").map(([b, cap]) => `${BUCKET_LABEL[b] ?? b}: ${resp.actions.filter((a) => a.capacity_bucket === b).length} of ${cap}`);
    return {
      text: `${when}: ${used} of ${resp.capacity.call} calls are used. By unit:`,
      bullets: byBucket,
      goto: { screen: "careteam", label: "Move the capacity slider", date: day?.date },
    };
  }

  if (has(q, "oxygen", "ventilator", "backup", "power", "outage", "equipment", "concentrator")) {
    const outageDays = ctx.days.filter((d) => d.outage >= 0.2);
    const plans = ctx.week.reduce((s, r) => s + (r?.actions.filter((a) => a.action === "backup_power_plan" || a.action === "cold_chain_plan").length ?? 0), 0);
    return {
      text: outageDays.length ? `Outages are forecast on ${outageDays.map((d) => fmtDate(d.date)).join(", ")}, up to ${Math.round(Math.max(...outageDays.map((d) => d.outage)) * 100)}% of ZIPs. ${fmtInt(plans)} backup-power and cold-chain plans are queued this week.` : "No outage is forecast in this window.",
      bullets: [
        "HHS emPOWER counts 36,146 electricity-dependent Medicare beneficiaries in NYC, 3,165 on oxygen.",
        "Rule of the list: a care-team call within two hours of an outage for anyone on powered equipment.",
      ],
      resources: searchLibrary("outage oxygen", "outage"),
      goto: { screen: "careteam", label: "Open the list", date: day?.date },
    };
  }

  if (has(q, "when", "landfall", "countdown", "how long", "days until", "arrive")) {
    const fl = ctx.days.findIndex((d) => d.flood);
    const ht = ctx.days.findIndex((d) => d.heat);
    const parts = [];
    if (fl >= 0) parts.push(`landfall ${fl === 0 ? "is today" : `is ${fmtDate(ctx.days[fl].date)}, in ${fl} day${fl === 1 ? "" : "s"}`}, with ${ctx.days[fl].surge.toFixed(0)} ft of surge into the evacuation zones`);
    if (ht >= 0) parts.push(`the heat alert starts ${fmtDate(ctx.days[ht].date)} with a heat index near ${Math.round(ctx.days[ht].maxHeatIndex)} °F`);
    return { text: parts.length ? `From today, ${fmtDate(ctx.days[0].date)}: ${parts.join("; ")}.` : "No alert-grade event is forecast in the seven-day window.", goto: { screen: "forecast", label: "Open the forecast" } };
  }

  if (has(q, "verify", "really from", "scam", "fraud", "trust", "phrase", "vsafe")) {
    return {
      text: "Every message goes only through VA channels the veteran already uses, carries a four-word verification phrase the caller reads back, says the VA will never ask you to pay, wire money or share bank details, and includes VSAFE 833-388-7233 and the Veterans Crisis Line, 988 press 1.",
      resources: searchLibrary("scam verify", "all"),
      goto: { screen: "message", label: "See a message with its checklist" },
    };
  }

  // Knowledge questions: the library answers, best match first.
  const hits = searchLibrary(q);
  if (hits.length > 0) {
    const top = hits[0];
    return { text: `${top.title}. ${top.summary}`, bullets: top.points.slice(0, 4), resources: hits.slice(0, 3), goto: { screen: "library", label: "Open the resource library" } };
  }

  return {
    text: "I answer from the numbers on this screen and the resource library, without a model call. Try one of these:",
    bullets: SUGGESTED,
  };
}

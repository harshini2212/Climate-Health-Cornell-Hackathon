/** Plain-language labels shared by every screen. The UI never shows a raw enum. */

export const NEED_LABEL: Record<string, string> = {
  breathing: "Breathing flare-up",
  heat: "Heat illness",
  mental: "Mental-health crisis",
  treatment_gap: "Treatment or medication gap",
  access_loss: "Loss of access to care",
};

export const NEED_SHORT: Record<string, string> = {
  breathing: "Breathing",
  heat: "Heat",
  mental: "Mental health",
  treatment_gap: "Treatment gap",
  access_loss: "Access to care",
};

export const ACTION_LABEL: Record<string, string> = {
  care_team_call: "Care-team call",
  check_in_call: "3-minute check-in",
  backup_power_plan: "Backup-power plan",
  cold_chain_plan: "Cold-chain plan",
  early_refill: "Early refill",
  switch_to_local_pickup: "Switch to local pickup",
  cooling_center_ride: "Cooling-center ride",
  clean_air_room: "Clean-air room",
  alt_site_booking: "Alternate-site booking",
  evacuation_assist: "Evacuation assist",
  assign_buddy: "Assign buddy",
  pharmacist_med_review: "Pharmacist review",
  controlled_substance_bridge: "Controlled-substance bridge",
  heap_application: "HEAP application",
  verified_text: "Verified text",
};

export const TIER_LABEL: Record<string, string> = {
  act_now: "Act now",
  find_out: "Find out",
  self_serve: "Self-serve",
  everyday: "Everyday",
};

/** Tier -> the template's risk-badge class. Act now is critical, everyday is quiet. */
export const TIER_BADGE: Record<string, string> = {
  act_now: "rb rb-critical",
  find_out: "rb rb-medium",
  self_serve: "rb rb-info",
  everyday: "rb rb-quiet",
};

export const OWNER_LABEL: Record<string, string> = {
  care_team: "Care team",
  pharmacist: "Pharmacist",
  partner: "Partner",
  automated: "Automated",
};

export const BUCKET_LABEL: Record<string, string> = {
  call: "calls",
  refill: "refills",
  ride: "rides",
  booking: "bookings",
  evac: "evacuation slots",
  partner_slot: "partner slots",
  pharmacist_slot: "pharmacist slots",
  va_fill: "VA fills",
  free: "texts",
};

export const BASELINE_LABEL: Record<string, string> = {
  leeward: "Leeward",
  rank_by_age: "Oldest first",
  rank_by_chronic: "Most conditions first",
  random: "Random",
};

export const RUNG_LABEL: Record<number, string> = {
  0: "Rung 0 · prior-only",
  1: "Rung 1 · pooled NUTS",
  2: "Rung 2 · interactions + SiteDown",
  3: "Rung 3 · ICAR + latent dose",
};

const DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** ISO date -> local Date, so a column never slips a day across time zones. */
export function parseDay(iso: string): Date {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, (m ?? 1) - 1, d ?? 1);
}
export const dayName = (iso: string) => DOW[parseDay(iso).getDay()];
export const dayLabel = (iso: string) => {
  const d = parseDay(iso);
  return `${d.getDate()} ${MON[d.getMonth()]}`;
};
/** "Mon 3 Aug" */
export const fmtDate = (iso: string) => `${dayName(iso)} ${dayLabel(iso)}`;

export const fmtInt = (n: number) => n.toLocaleString("en-US");
export const fmt1 = (n: number) => n.toFixed(1);
export const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

/**
 * "W.O. · 4471". A board on a wall in a shared clinical space shows a handle, never a
 * name. The full card is one click away for the person who needs it.
 */
export function handleFor(nameDisplay: string, veteranId: string): string {
  const initials = nameDisplay.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]?.toUpperCase()).join(".");
  const digits = veteranId.replace(/\D/g, "").slice(-4) || veteranId.slice(-4);
  return `${initials ? `${initials}.` : "—"} · ${digits}`;
}

export const initialsOf = (name: string) =>
  name.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]?.toUpperCase()).join("");

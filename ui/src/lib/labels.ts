/** Plain-language labels shared by every screen. */

export const NEED_LABEL: Record<string, string> = {
  breathing: "Breathing flare-up",
  heat: "Heat illness",
  mental: "Mental-health crisis",
  treatment_gap: "Treatment or medication gap",
  access_loss: "Loss of access to care",
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

export const BASELINE_LABEL: Record<string, string> = {
  leeward: "Leeward",
  rank_by_age: "Oldest first",
  rank_by_chronic: "Most conditions first",
  random: "Random",
};

export const RUNG_LABEL: Record<number, string> = {
  0: "rung 0 · prior-only",
  1: "rung 1 · pooled NUTS",
  2: "rung 2 · interactions + SiteDown",
  3: "rung 3 · ICAR + latent dose",
};

export function fmtDate(iso: string): string {
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
}

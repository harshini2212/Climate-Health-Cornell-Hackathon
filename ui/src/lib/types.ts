/**
 * TypeScript mirrors of leeward/api/schemas.py. Those shapes are frozen; if a field here
 * disagrees with the pydantic model, the pydantic model wins and this file is wrong.
 * tests/test_ui_fixtures.py parses every fixture through the pydantic models so the two
 * cannot drift silently.
 */

export const NEEDS = ["breathing", "heat", "mental", "treatment_gap", "access_loss"] as const;
export type Need = (typeof NEEDS)[number];

export const TIERS = ["act_now", "find_out", "self_serve", "everyday"] as const;
export type Tier = (typeof TIERS)[number];

export type Capacity = Record<string, number>;

/** leeward.schema.DEFAULT_CAPACITY. The slider moves `call`. */
export const DEFAULT_CAPACITY: Capacity = {
  call: 40,
  refill: 200,
  ride: 15,
  booking: 20,
  evac: 8,
  partner_slot: 10,
  pharmacist_slot: 12,
  va_fill: 30,
  free: 10000,
};

// GET /forecast ------------------------------------------------------------

export interface ZipHazard {
  modzcta: string;
  date: string;
  heat_index_max_f: number;
  hot_day: boolean;
  heat_alert: boolean;
  pm25: number;
  smoke_alert: boolean;
  flood_warning: boolean;
  flash_flood_emergency: boolean;
  surge_ft: number;
  evac_zone_ordered: number;
  outage_frac: number;
  mail_delivery_disrupted: boolean;
}

export interface FacilityStatus {
  facility_id: string;
  name: string;
  lat: number;
  lon: number;
  evac_zone: number;
  site_down: boolean;
  site_dependent_services: boolean;
}

export interface ForecastResponse {
  scenario: string;
  day: number;
  dates: string[];
  zips: ZipHazard[];
  facilities: FacilityStatus[];
  headline?: string | null;
}

// GET /scores --------------------------------------------------------------

export interface ZipScore {
  modzcta: string;
  need: string;
  expected_count: number;
  lo80: number;
  hi80: number;
  n_panel: number;
}

export interface ScoresResponse {
  date: string;
  need: string;
  zips: ZipScore[];
  facilities: ZipScore[];
  model_rung: number;
}

// POST /actions ------------------------------------------------------------

export interface ActionsRequest {
  date: string;
  capacity: Capacity;
  group_floor?: Record<string, number> | null;
  prior_scale?: number;
  scenario?: string;
}

export interface ActionRow {
  action_id: string;
  rank: number;
  veteran_id: string;
  name_display: string;
  modzcta: string;
  borough: string;
  action: string;
  tier: Tier;
  eha: number;
  capacity_bucket: string;
  owner: string;
  /** Card-safe: urgency and timing only. The one reason line the wall board may show. */
  headline: string;
  /** Names the service and the driver. Drill-down screens only, never the board. */
  rationale: string;
  /** Names a condition or a medicine. Drill-down screens only, never the board. */
  top_driver?: string | null;
  message_id?: string | null;
}

export interface BaselineResult {
  name: string;
  total_eha: number;
}

export interface ActionsResponse {
  date: string;
  capacity: Capacity;
  actions: ActionRow[];
  total_eha: number;
  baselines: BaselineResult[];
  n_panel: number;
  n_selected: number;
  counts_by_tier: Record<string, number>;
  model_rung: number;
}

// GET /veteran/{id} --------------------------------------------------------

export interface NeedScore {
  need: string;
  p_mean: number;
  p_lo80: number;
  p_hi80: number;
  p_epistemic_share: number;
  drivers: string[];
  driver_contribs: number[];
}

export interface MedicationFlags {
  n_active_meds: number;
  thermoreg_score: number;
  acb_score: number;
  combo_raas_diuretic: boolean;
  renal_triple: boolean;
  cold_chain: boolean;
  controlled: boolean;
  narrow_ti: boolean;
  mail_order_pharmacy: boolean;
  days_supply_remaining: number;
  notes: string[];
}

export interface VeteranCard {
  veteran_id: string;
  name_display: string;
  age: number;
  modzcta: string;
  borough: string;
  facility_id: string;
  facility_name: string;
  date: string;
  tier: Tier;
  why_this_tier: string;
  needs: NeedScore[];
  medications: MedicationFlags;
  conditions: string[];
  powered_equipment: string;
  caregiver: string;
  floor: string;
  evac_zone: number;
  planned_actions: string[];
  is_synthetic: boolean;
}

// GET /message/{action_id} -------------------------------------------------

export interface Message {
  message_id: string;
  action_id: string;
  veteran_id: string;
  channel: string;
  addressed_to: string;
  verification_phrase: string;
  body: string;
  includes_never_pay_line: boolean;
  includes_vsafe: boolean;
  includes_crisis_line: boolean;
  scam_card_url?: string | null;
}

// GET /report --------------------------------------------------------------

export interface RecoveryRow {
  parameter: string;
  truth: number;
  post_mean: number;
  lo90: number;
  hi90: number;
  covered: boolean;
}

export interface CalibrationBin {
  need: string;
  predicted: number;
  observed: number;
  n: number;
}

export interface FairnessRow {
  stratum: string;
  group: string;
  n: number;
  ece: number;
  fnr: number;
  fnr_ratio_to_cohort: number;
  flagged: boolean;
}

export interface AblationRow {
  dropped: string;
  ece: number;
  harm_averted_at_40: number;
}

export interface DecisionQualityRow {
  k: number;
  strategy: string;
  harm_averted: number;
}

/**
 * leeward/eval/discrimination.py, one row per need on the held-out window.
 *
 * `within_day_auc` is the headline and `pooled_auc` is the flattering one: the call list is
 * chosen within a day, so pooled AUC includes credit for knowing today is a heat wave, which
 * the team cannot act on. Show both or neither. Nulls mean "no events to rank", not zero.
 */
export interface DiscriminationRow {
  need: string;
  within_day_auc?: number | null;
  pooled_auc?: number | null;
  pr_auc?: number | null;
  /** Brier skill score: 0 for a constant at the base rate, 1 for perfect, negative for
   *  worse than knowing nothing. The score on which the constant does NOT beat us. */
  scaled_brier?: number | null;
  brier?: number | null;
  lift_at_1pct?: number | null;
  lift_at_10pct?: number | null;
  /** min(1/q, 1/base_rate) — the best lift@1% anything could score. Lift without its
   *  ceiling is uninterpretable, so the screen never shows one without the other. */
  lift_ceiling_1pct?: number | null;
  base_rate: number;
  n: number;
  n_events: number;
  /** Days that could be scored, and days there were. Eventless days have no ranking to
   *  score and are dropped; that conditions on the outcome, so both numbers are shown. */
  n_days: number;
  n_days_total?: number;
}

export interface ReportResponse {
  model_rung: number;
  rhat_max?: number | null;
  divergences?: number | null;
  recovery: RecoveryRow[];
  recovery_coverage?: number | null;
  calibration: CalibrationBin[];
  ece_by_need: Record<string, number>;
  /** What a single number at each need's base rate scores on the same bins: 0, by
   *  construction. The control for `ece_by_need`, rendered beside it and never hidden. */
  constant_ece?: Record<string, number>;
  discrimination?: DiscriminationRow[];
  ablations: AblationRow[];
  decision_quality: DecisionQualityRow[];
  fairness: FairnessRow[];
  fairness_failed: boolean;
  generated_at?: string | null;
}

/**
 * Chart color, by the job it does. Values are from the validated reference palette
 * (dataviz skill, references/palette.md); nothing here is eyeballed.
 *
 * - sequential: one hue (blue), light -> dark, for magnitude on the choropleth
 * - status: reserved meanings, always paired with an icon or label, never a series
 * - tiers are ordinal (urgency), so they take steps of the same blue ramp
 */

export type RGBA = [number, number, number, number];

const hex = (h: string, a = 255): RGBA => [
  parseInt(h.slice(1, 3), 16),
  parseInt(h.slice(3, 5), 16),
  parseInt(h.slice(5, 7), 16),
  a,
];

/** Sequential blue, steps 100 -> 700. Index 0 means "near zero" and recedes. */
export const SEQUENTIAL_HEX = [
  "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
  "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
];
export const SEQUENTIAL: RGBA[] = SEQUENTIAL_HEX.map((h) => hex(h));

/** Map a value in [0, max] to a ramp step. */
export function rampIndex(value: number, max: number): number {
  if (!(max > 0) || !(value > 0)) return 0;
  const t = Math.min(value / max, 1);
  return Math.min(SEQUENTIAL.length - 1, Math.floor(t * (SEQUENTIAL.length - 1) + 0.5));
}

export const STATUS_HEX = {
  good: "#0ca30c",
  warning: "#fab219",
  serious: "#ec835a",
  critical: "#d03b3b",
} as const;
export const STATUS: Record<keyof typeof STATUS_HEX, RGBA> = {
  good: hex(STATUS_HEX.good),
  warning: hex(STATUS_HEX.warning),
  serious: hex(STATUS_HEX.serious),
  critical: hex(STATUS_HEX.critical),
};

export const NO_DATA: RGBA = [225, 224, 217, 255]; // gridline hairline gray
export const SERIES_1 = "#2a78d6";
export const SERIES_1_DARK = "#3987e5";

/** Facility marker: open sites take the series hue; a closed site is status-critical. */
export const FACILITY_OPEN: RGBA = hex("#2a78d6");
export const FACILITY_DOWN: RGBA = hex(STATUS_HEX.critical);

/** Tier is ordinal: act now is darkest, everyday is the neutral ink. */
export const TIER_STEP: Record<string, string> = {
  act_now: "#1c5cab",
  find_out: "#3987e5",
  self_serve: "#86b6ef",
  everyday: "#c3c2b7",
};
export const TIER_LABEL: Record<string, string> = {
  act_now: "Act now",
  find_out: "Find out",
  self_serve: "Self-serve",
  everyday: "Everyday",
};

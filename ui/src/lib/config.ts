/**
 * Demo configuration. One scenario, one demo day, used by every screen so the ribbon, the
 * map, the list and the card all open on the same morning.
 *
 * `day` is the scenario-day index the API's /forecast takes (0 = the scenario's first
 * day). For Sandy-then-heat, day 63 is landfall: the coastal flood warning, the surge into
 * evacuation zones 1-2, the outage, and station 630 closing, with the heat wave three days
 * later. Opening on day 0 shows an empty ribbon, which is the wrong first impression.
 */

export interface ScenarioMeta {
  key: string;
  label: string;
  day: number;
  /** One line for the topbar. */
  context: string;
  /** Scored by the model right now? Only Sandy has scores.parquet behind it. */
  scored: boolean;
}

export const SCENARIOS: ScenarioMeta[] = [
  { key: "sandy_then_heat", label: "Sandy, then heat", day: 63, context: "Landfall Mon 3 Aug · heat wave from Thu 6 Aug", scored: true },
  { key: "ida_flash_flood", label: "Ida flash flood", day: 78, context: "Flash flood emergency Wed 1 Sep 2021", scored: false },
  { key: "smoke_2023", label: "Smoke, June 2023", day: 51, context: "AirNow replay, peak Wed 7 Jun 2023", scored: false },
];

export const DEFAULT_SCENARIO = SCENARIOS[0];

export function scenarioMeta(key: string): ScenarioMeta {
  return SCENARIOS.find((s) => s.key === key) ?? DEFAULT_SCENARIO;
}

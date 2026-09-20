/**
 * The scenario the demo runs. Which *day* it opens on is not typed here: `lib/demo.ts`
 * asks the API for the window two days before the first alert (leeward/demo.py), and the
 * topbar can switch to the calm week. Only Sandy has scores.parquet behind it right now.
 */

export interface ScenarioMeta {
  key: string;
  label: string;
  /** One line for the topbar until the forecast has said which day it opens on. */
  context: string;
}

export const DEFAULT_SCENARIO: ScenarioMeta = {
  key: "sandy_then_heat",
  label: "Sandy, then heat",
  context: "Landfall Mon 3 Aug · heat wave from Thu 6 Aug",
};

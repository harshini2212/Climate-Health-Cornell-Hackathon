/**
 * Where the demo opens.
 *
 * The scenario is 120 days long and almost all of them are quiet, so the UI asks the API
 * for no day at all and takes the window it is given: `leeward/demo.py` picks the one two
 * days in front of the first alert, off the hazards table, so it stays right when the
 * dates move. `undefined` here means exactly that — "you choose" — and it is the default.
 *
 * `make demo DAY=0` pins it instead, and the topbar switches between the two live. Day 0
 * is the calm week: "here is a calm week, and here is the same team three days later".
 */

/** `make demo DAY=<n>` → VITE_DEMO_DAY. Unset, empty or nonsense means "let the API pick". */
export const PINNED_DAY: number | undefined = (() => {
  const raw = import.meta.env.VITE_DEMO_DAY;
  if (raw === undefined || raw === null || `${raw}`.trim() === "") return undefined;
  const n = Number(raw);
  return Number.isInteger(n) && n >= 0 ? n : undefined;
})();

export interface DayChoice {
  key: string;
  label: string;
  /** undefined = ask the API for its opening window. */
  day: number | undefined;
}

/** The two beats, in the order the demo tells them. */
export const DAY_CHOICES: DayChoice[] = [
  { key: "opening", label: "Landfall week", day: undefined },
  { key: "calm", label: "Day 0 · calm week", day: 0 },
];

/** The choice `make demo` booted with: a pinned DAY if there was one, else the opening week. */
export const INITIAL_CHOICE: string =
  PINNED_DAY === undefined ? "opening" : PINNED_DAY === 0 ? "calm" : "pinned";

/** DAY=7, say, is neither of the two named beats, so it gets its own entry in the picker. */
export const CHOICES: DayChoice[] =
  INITIAL_CHOICE === "pinned"
    ? [{ key: "pinned", label: `Day ${PINNED_DAY} · make demo DAY`, day: PINNED_DAY }, ...DAY_CHOICES]
    : DAY_CHOICES;

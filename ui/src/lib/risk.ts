/**
 * The risk a queue row is actually claiming, pulled back out of its rationale.
 *
 * `ActionRow` carries the number as prose in `rationale` and nowhere as fields, and
 * `api/schemas.py` is frozen mid-build, so until it grows `p_mean`/`p_lo80`/`p_hi80` the
 * only way a list row can show its interval is to read the sentence the decision layer
 * already wrote. That sentence is generated in `leeward/decision/`, not typed by a person,
 * so its shape is stable -- but a parser over prose is still a parser over prose, and
 * every caller treats `null` as "show the sentence instead", never as "assume it is fine".
 *
 * Two shapes exist, and the second one is the whole point of the Find-out tier:
 *   act now   "...: 55% chance of a gap in treatment (80% interval 12%-95%), driven by ..."
 *   find out  "...: 24% chance of heat illness, and 32% of the uncertainty is what we do
 *              not know about them."
 * A Find-out row states no interval because that tier exists for the rows whose interval
 * is too wide to act on. What it reports instead is the epistemic share: the part of the
 * uncertainty a three-minute phone call could actually remove.
 */

export interface RowRisk {
  /** Posterior mean for this row's single worst need, as a fraction of 1. */
  p: number;
  /** 80% credible interval, when the rationale states one. */
  lo?: number;
  hi?: number;
  /** Share of the variance that is reducible by asking. Find-out rows only. */
  epistemic?: number;
}

const P = /(\d+)%\s+chance/;
const INTERVAL = /80%\s+interval\s+(\d+)%\s*[-–]\s*(\d+)%/;
const EPISTEMIC = /(\d+)%\s+of the uncertainty/;

/**
 * Read one rationale sentence. Returns null when the shape is not one of the two known
 * ones, which is the signal to fall back to the prose rather than draw a bar over a guess.
 */
export function parseRisk(rationale?: string | null): RowRisk | null {
  if (!rationale) return null;
  const p = P.exec(rationale);
  if (!p) return null;
  const out: RowRisk = { p: Number(p[1]) / 100 };
  const iv = INTERVAL.exec(rationale);
  if (iv) {
    const lo = Number(iv[1]) / 100;
    const hi = Number(iv[2]) / 100;
    // A pair that does not bracket the mean would draw a bar running backwards; drop it.
    if (lo <= out.p && out.p <= hi) {
      out.lo = lo;
      out.hi = hi;
    }
  }
  const ep = EPISTEMIC.exec(rationale);
  if (ep) out.epistemic = Number(ep[1]) / 100;
  return out;
}

/** "12-95%", the half of the sentence the table used to clip. Empty when there is none. */
export function intervalText(r: RowRisk): string {
  return r.lo === undefined || r.hi === undefined
    ? ""
    : `${Math.round(r.lo * 100)}–${Math.round(r.hi * 100)}%`;
}

/** How wide that interval is, in percentage points -- the number that sorts Find-out. */
export function intervalWidth(r: RowRisk): number | null {
  return r.lo === undefined || r.hi === undefined ? null : r.hi - r.lo;
}

/**
 * The headline is always "<need>, N% chance — <when>", written by the decision layer.
 * Split it so a table can put the need beside the action and the timing beside the tier,
 * instead of repeating a percentage the interval bar is already showing.
 */
export function splitHeadline(headline?: string | null): { need: string; when: string } {
  if (!headline) return { need: "", when: "" };
  const [head, ...rest] = headline.split("—");
  const when = rest.join("—").trim();
  const need = head.split(",")[0].trim();
  return { need, when };
}

/**
 * What is left of the rationale once the bar has taken the number, so a row never prints
 * the same percentage twice. "Driven by X" is the clause worth keeping; a Find-out row
 * keeps its own sentence, which is about what is unknown rather than what is driving it.
 */
export function rationaleTail(rationale: string): string {
  const driven = /,\s*driven by\s+(.+?)\.?\s*$/i.exec(rationale);
  if (driven) return `Driven by ${driven[1]}`;
  const ep = /,\s*and\s+(\d+% of the uncertainty.+?)\.?\s*$/i.exec(rationale);
  if (ep) return ep[1].charAt(0).toUpperCase() + ep[1].slice(1);
  return rationale;
}

/**
 * One row's risk, as the interval it actually is.
 *
 * The veteran card already draws a need this way (`NeedBar`), but it scales each card to
 * its own worst need, which is right for reading one person and wrong for reading a list:
 * a clinician scanning a queue is comparing rows, so every bar here shares one fixed
 * 0-100% axis. Same grammar as the card -- band is the 80% credible interval, the tick is
 * the posterior mean, the hatch is the part of the spread that is model uncertainty --
 * so the two screens teach each other rather than competing.
 *
 * At rung 0 (prior-only) these bands are enormous, median 42 points wide. That is not a
 * rendering problem to be smoothed over; it is the reason the Find-out tier exists, and
 * the list should say so on every row rather than print a confident-looking "55%".
 */

import type { RowRisk } from "../lib/risk";
import { intervalText } from "../lib/risk";

const pct = (v: number) => `${Math.round(v * 100)}%`;

export function RiskBar({ r, need }: { r: RowRisk; need?: string }) {
  const hasInterval = r.lo !== undefined && r.hi !== undefined;
  const lo = r.lo ?? r.p;
  const hi = r.hi ?? r.p;
  const left = lo * 100;
  const width = Math.max(0.8, (hi - lo) * 100);
  const meanAt = r.p * 100;
  // The epistemic slice is centred on the mean, the way the veteran card draws it.
  const epiWidth = hasInterval ? width * Math.min(1, Math.max(0, r.epistemic ?? 0)) : 0;
  const epiLeft = Math.min(Math.max(meanAt - epiWidth / 2, left), left + width - epiWidth);

  const label = hasInterval
    ? `${pct(r.p)}, 80% credible interval ${intervalText(r)}`
    : `${pct(r.p)}, no interval stated`;

  return (
    <div className="ivcell">
      <div className="ivhead">
        <b>{pct(r.p)}</b>
        {hasInterval ? (
          <small className="muted">{intervalText(r)}</small>
        ) : (
          <small className="muted">interval not stated</small>
        )}
      </div>
      <div className="ivtrack" role="img" aria-label={label} title={label}>
        {hasInterval && <div className="ivb" style={{ left: `${left}%`, width: `${width}%` }} />}
        {epiWidth > 0 && <div className="ive" style={{ left: `${epiLeft}%`, width: `${epiWidth}%` }} />}
        <div className="ivm" style={{ left: `${meanAt}%` }} />
      </div>
      {need && <div className="ivfoot">{need}</div>}
    </div>
  );
}

/**
 * The header a column of these needs, because a bar with no axis is a decoration. Said
 * once above the table rather than on every row.
 */
export function RiskAxis() {
  return (
    <span className="sub ivaxis">
      <span>0%</span>
      <span>50%</span>
      <span>100%</span>
    </span>
  );
}

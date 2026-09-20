/**
 * Small, dependency-free charts in the template's chart style. One axis, one hue per
 * chart; a dashed red line for a clinical threshold; labels in the muted axis ink.
 */

import { fmtDate } from "../lib/labels";

interface SparkProps {
  dates: string[];
  values: number[];
  threshold?: number;
  unit?: string;
  height?: number;
}

/** A single-series line with optional threshold. Points carry a native tooltip. */
export function Spark({ dates, values, threshold, unit = "", height = 112 }: SparkProps) {
  const W = 420, H = height, padL = 36, padR = 10, padT = 12, padB = 22;
  if (values.length === 0) return null;
  const lo = Math.min(...values, threshold ?? Infinity) * 0.95;
  const hi = (Math.max(...values, threshold ?? -Infinity) * 1.06) || 1;
  const x = (i: number) => padL + (i * (W - padL - padR)) / Math.max(1, values.length - 1);
  const y = (v: number) => padT + (H - padT - padB) * (1 - (v - lo) / (hi - lo));
  const path = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const area = `${path} L${x(values.length - 1).toFixed(1)},${(H - padB).toFixed(1)} L${x(0).toFixed(1)},${(H - padB).toFixed(1)} Z`;
  return (
    <svg className="chart2" viewBox={`0 0 ${W} ${H}`} role="img">
      <line x1={padL} x2={W - padR} y1={H - padB} y2={H - padB} stroke="var(--line)" />
      {threshold !== undefined && (
        <>
          <line x1={padL} x2={W - padR} y1={y(threshold)} y2={y(threshold)} stroke="var(--red)" strokeDasharray="4 4" strokeWidth="1" />
          <text className="axl" x={W - padR} y={y(threshold) - 4} textAnchor="end" style={{ fill: "var(--red)" }}>{threshold}{unit}</text>
        </>
      )}
      <path d={area} fill="var(--accentbg)" />
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinejoin="round" />
      {values.map((v, i) => (
        <circle key={i} cx={x(i)} cy={y(v)} r={3.5} fill="var(--panel)" stroke="var(--accent)" strokeWidth="2">
          <title>{fmtDate(dates[i])}: {v.toFixed(0)}{unit}</title>
        </circle>
      ))}
      {dates.map((d, i) => (
        <text key={d} className="axl" x={x(i)} y={H - 6} textAnchor="middle">{fmtDate(d).split(" ")[0]}</text>
      ))}
      <text className="axl" x={padL - 6} y={y(hi) + 4} textAnchor="end">{hi.toFixed(0)}</text>
      <text className="axl" x={padL - 6} y={y(lo) + 4} textAnchor="end">{lo.toFixed(0)}</text>
    </svg>
  );
}

interface HBarsProps {
  rows: { label: string; value: number; lead?: boolean }[];
  max?: number;
  format?: (v: number) => string;
}

interface DotRowsProps {
  rows: { label: string; value: number; lead?: boolean }[];
  format?: (v: number) => string;
  /** Unit for the delta column, e.g. "need-days". */
  unit?: string;
}

/**
 * Four strategies whose totals differ by under one percent.
 *
 * Bars from zero cannot show that: at 220.0 against 218.3 every bar is the same bar, and
 * a reader concludes either "identical" or "the chart is broken". Bars with the zero cut
 * off are worse -- they turn a 0.8% gap into a landslide. For values this close the form
 * is a dot plot on an axis that states its own range, with the lead series in the accent
 * and the rest in grey, and the gap written out in the unit it is measured in.
 *
 * The honest reading, which this makes visible rather than hides: at rung 0 the ranking
 * is barely ahead of calling the oldest first. That is what a prior-only model does, and
 * the Model report's realized harm-averted is the number that separates them.
 */
export function DotRows({ rows, format = (v) => v.toFixed(1), unit = "" }: DotRowsProps) {
  const vals = rows.map((r) => r.value);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const span = hi - lo || 1;
  // A tenth of the spread of padding at each end, so no dot sits on the axis edge.
  const from = lo - span * 0.18;
  const to = hi + span * 0.18;
  const at = (v: number) => (100 * (v - from)) / (to - from);
  const lead = rows.find((r) => r.lead) ?? rows[0];

  return (
    <div className="dotplot">
      {rows.map((r) => {
        const delta = r.value - lead.value;
        return (
          <div key={r.label} className={`dotrow${r.lead ? " lead" : ""}`}>
            <span className="dl">{r.label}</span>
            <div className="dt">
              <div className="dax" />
              <span className="dd" style={{ left: `${at(r.value)}%` }} title={`${r.label}: ${format(r.value)}`} />
            </div>
            <span className="dv">{format(r.value)}</span>
            <span className="dg">{r.lead ? "—" : `${delta > 0 ? "+" : ""}${delta.toFixed(2)}`}</span>
          </div>
        );
      })}
      <div className="dfoot">
        <span />
        <span className="dscale"><span>{format(from)}</span><span>{format(to)}</span></span>
        <span className="dnote">gap to {lead.label.toLowerCase()}{unit ? `, ${unit}` : ""}</span>
      </div>
    </div>
  );
}

/** Nominal categories share one hue; the lead row is set apart by weight, not color. */
export function HBars({ rows, max, format = (v) => v.toFixed(1) }: HBarsProps) {
  const top = Math.max(max ?? 0, ...rows.map((r) => r.value), 1e-9);
  return (
    <div className="hbars">
      {rows.map((r) => (
        <div key={r.label} className={`hbar${r.lead ? " lead" : ""}`}>
          <span className="hl">{r.label}</span>
          <div className="ht"><i style={{ width: `${(100 * r.value) / top}%` }} /></div>
          <span className="hv">{format(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

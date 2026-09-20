import { useEffect, useMemo, useState } from "react";
import { IconAlert, IconShield } from "../components/Icons";
import { getReport, lastSource, type Source } from "../lib/api";
import { STRATEGY_STEP } from "../lib/colors";
import { BASELINE_LABEL, EHA_REALIZED, NEED_LABEL, NEED_SHORT, RUNG_LABEL, fmtInt } from "../lib/labels";
import {
  NEEDS,
  type CalibrationBin,
  type DecisionQualityRow,
  type RecoveryRow,
  type ReportResponse,
} from "../lib/types";

/**
 * The model report: the four claims this project is allowed to make on stage, each one
 * drawn from `report/report.json` exactly as `make report` wrote it.
 *
 *   1. recovery      — does the fit find the world the simulator built?
 *   2. reliability   — when it says 3 %, does it happen 3 % of the time?
 *   3. harm averted  — does the ranking beat oldest-first, most-conditions-first, random?
 *   4. fairness      — do the misses fall evenly across groups?
 *
 * Titles and subtitles are the ones `leeward/eval/*.py` puts on the offline Plotly charts,
 * so the screen and `report/*.html` cannot end up claiming different things.
 *
 * The fairness table renders whether or not it passes, and flagged rows are marked. That
 * is a project rule; `tests/test_ui_fixtures.py` holds this screen to it and
 * `tests/test_guardrails.py` holds the backend to the other half.
 */

/** leeward/eval/fairness.py's FNR_GAP. The ratio column draws the band it tests. */
const FNR_GAP = 0.2;

/** leeward/eval/fairness.py's CALL_BUDGET, itself DEFAULT_CAPACITY["call"]. The coverage
 *  column counts events that got one of these, so the label has to name the same number. */
const CALL_BUDGET = 40;

const NUM = (v: number, d = 2) => v.toFixed(d);
const PCT = (v: number, d = 1) => `${(100 * v).toFixed(d)}%`;

/** "beta_copd (breathing)" -> "breathing". Every recovery row names its need this way. */
function needOf(parameter: string): string {
  const m = /\(([^)]+)\)\s*$/.exec(parameter);
  return m ? m[1] : "other";
}

/** "2026-09-20 03:50 UTC". Not toLocaleString: the same run must read the same on stage. */
function fmtRun(iso: string | null | undefined): string {
  if (!iso) return "never run";
  const [day, rest] = iso.split("T");
  if (!rest) return iso;
  return `${day} ${rest.slice(0, 5)}${/(Z|\+00:00)$/.test(rest) ? " UTC" : ""}`;
}

/**
 * An empty section, in the redesign's `.pending` block. Every one of them says the same
 * thing in the same words: nothing ran here. An empty table is the absence of evidence and
 * must never read as a clean bill.
 */
function NotRun({ what, cmd, why }: { what: string; cmd: string; why: string }) {
  return (
    <div className="pending">
      <span>
        <b>{what}</b> · <code>{cmd}</code>
        <div style={{ marginTop: 3 }}>Empty means not run. {why}</div>
      </span>
    </div>
  );
}

// --------------------------------------------------------------------------- #
// 1. Recovery: truth against the 90 % interval
// --------------------------------------------------------------------------- #

/**
 * One parameter's interval, rescaled so every row is the same width, with the generating
 * truth as the dot. The intervals span three orders of magnitude across 60 parameters, so
 * a shared value axis would render most of them as a smudge; rescaling each to its own
 * half-width keeps the only question this chart asks readable: is the dot inside the bar?
 */
function Whisker({ row }: { row: RecoveryRow }) {
  const half = Math.max((row.hi90 - row.lo90) / 2, 1e-12);
  const z = (row.truth - row.post_mean) / half;
  // The viewBox matches the rendered box 1:1 (see .whisk), so the truth dot stays a circle
  // instead of stretching into an ellipse the way a scaled viewBox would.
  const x = (u: number) => 88 + Math.max(-1.48, Math.min(1.48, u)) * 56;
  const ink = row.covered ? "var(--accent)" : "var(--crit)";
  return (
    <svg className="whisk" viewBox="0 0 176 14" aria-hidden="true">
      <line x1="3" x2="173" y1="7" y2="7" stroke="var(--line2)" strokeWidth="1" />
      <line x1={x(-1)} x2={x(1)} y1="7" y2="7" stroke="var(--accentbd)" strokeWidth="6" strokeLinecap="round" />
      <line x1={x(-1)} x2={x(-1)} y1="2.5" y2="11.5" stroke="var(--accent)" strokeWidth="1.2" />
      <line x1={x(1)} x2={x(1)} y1="2.5" y2="11.5" stroke="var(--accent)" strokeWidth="1.2" />
      <line x1="88" x2="88" y1="3.5" y2="10.5" stroke="var(--muted)" strokeWidth="1" />
      <circle cx={x(z)} cy="7" r="3.4" fill={ink} stroke="var(--panel)" strokeWidth="1" />
    </svg>
  );
}

function Recovery({ report }: { report: ReportResponse }) {
  const [need, setNeed] = useState("all");
  const [missesOnly, setMissesOnly] = useState(false);
  const priorOnly = report.model_rung === 0;

  const rows = useMemo(() => {
    const keep = report.recovery.filter(
      (r) => (need === "all" || needOf(r.parameter) === need) && (!missesOnly || !r.covered));
    // Misses first: a claim about recovery is only worth as much as its worst row.
    return [...keep].sort((a, b) => Number(a.covered) - Number(b.covered));
  }, [report.recovery, need, missesOnly]);

  const missed = report.recovery.filter((r) => !r.covered).length;
  const coverage = report.recovery_coverage;

  if (!report.recovery.length) {
    return (
      <NotRun what="Parameter recovery has no rows in this report"
              cmd="python -m leeward.eval.recovery"
              why="It is not a claim that every parameter was recovered." />
    );
  }

  return (
    <div className="card">
      <div className="ch">
        <h3>{priorOnly ? "Prior coverage" : "Parameter recovery"}: truth against the 90% interval</h3>
        <span className="sp" />
        <span className={missed ? "pillbadge pill-watch" : "pillbadge pill-healthy"}>
          {coverage === null || coverage === undefined ? "—" : PCT(coverage, 0)} covered · bar is 90%
        </span>
      </div>
      <p className="lede" style={{ marginBottom: 10 }}>
        {priorOnly ? "rung 0 has no fit, so these are prior intervals" : "90% posterior intervals"}
        {" · "}{report.recovery.length - missed} of {report.recovery.length} parameters covered
        {" · "}{missed} outside. Every whisker is rescaled to the same width, so the dot is
        the generating truth in units of that parameter's own interval: inside the bar is a
        hit, outside is a miss.
      </p>
      <div className="ctrls">
        <span className={`iv ${need === "all" ? "on" : ""}`} onClick={() => setNeed("all")}>All needs</span>
        {NEEDS.map((n) => (
          <span key={n} className={`iv ${need === n ? "on" : ""}`} onClick={() => setNeed(n)}>
            {NEED_LABEL[n]}
          </span>
        ))}
        <span className={`iv ${missesOnly ? "on" : ""}`} onClick={() => setMissesOnly(!missesOnly)}>
          Misses only {missed ? `(${missed})` : ""}
        </span>
      </div>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>Parameter</th>
              <th style={{ width: 190 }}>truth vs interval</th>
              <th className="n">Truth</th>
              <th className="n">{priorOnly ? "Prior mean" : "Post. mean"}</th>
              <th className="n">90% interval</th>
              <th>In</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.parameter} className={r.covered ? "" : "missrow"}>
                <td>{r.parameter}</td>
                <td><Whisker row={r} /></td>
                <td className="n">{NUM(r.truth)}</td>
                <td className="n">{NUM(r.post_mean)}</td>
                <td className="n muted">[{NUM(r.lo90)}, {NUM(r.hi90)}]</td>
                <td>{r.covered
                  ? <span className="mk ok">✓</span>
                  : <span className="rb rb-critical">missed</span>}</td>
              </tr>
            ))}
            {!rows.length && (
              <tr><td colSpan={6} className="muted">No parameter matches that filter.</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="legend">
        <span><i style={{ background: "var(--accentbd)", height: 6 }} /> 90% interval, rescaled</span>
        <span><i style={{ background: "var(--muted)", height: 6, width: 3 }} /> interval centre</span>
        <span><i style={{ background: "var(--accent)", height: 8, width: 8, borderRadius: 8 }} /> truth, covered</span>
        <span><i style={{ background: "var(--crit)", height: 8, width: 8, borderRadius: 8 }} /> truth, missed</span>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- #
// 2. Reliability, per need
// --------------------------------------------------------------------------- #

interface LogScale { lo: number; hi: number }

/**
 * One need's reliability curve on a log-log square: predicted risk across, what actually
 * happened up, the diagonal for "exactly right". Log axes because a heat day runs at 10%
 * and a breathing day at 0.6%; on a linear 0–1 axis all five needs would sit in the corner.
 * The scale is shared by all five panels, so they can be read against each other.
 */
function Reliability({ need, bins, ece, scale }: {
  need: string; bins: CalibrationBin[]; ece: number | undefined; scale: LogScale;
}) {
  const W = 210, H = 210, padL = 34, padR = 10, padT = 10, padB = 28;
  const lg = (v: number) => Math.log10(Math.max(v, scale.lo));
  const span = lg(scale.hi) - lg(scale.lo);
  const x = (v: number) => padL + ((lg(v) - lg(scale.lo)) / span) * (W - padL - padR);
  const y = (v: number) => H - padB - ((lg(v) - lg(scale.lo)) / span) * (H - padT - padB);
  const pts = [...bins].sort((a, b) => a.predicted - b.predicted);
  const maxN = Math.max(1, ...pts.map((p) => p.n));
  const ticks = [0.002, 0.01, 0.05, 0.1].filter((t) => t >= scale.lo && t <= scale.hi);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p.predicted).toFixed(1)},${y(p.observed).toFixed(1)}`).join(" ");

  return (
    <div className="relpanel">
      <div className="relhead">
        <span className="rl" title={NEED_LABEL[need] ?? need}>{NEED_SHORT[need] ?? need}</span>
        <span className={ece !== undefined && ece > 0.03 ? "re red" : "re"}>
          ECE {ece === undefined ? "—" : NUM(ece, 4)}
        </span>
      </div>
      <svg className="chart2" viewBox={`0 0 ${W} ${H}`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={padT} y2={H - padB} stroke="var(--line2)" strokeWidth="1" />
            <line x1={padL} x2={W - padR} y1={y(t)} y2={y(t)} stroke="var(--line2)" strokeWidth="1" />
            <text className="axl" x={x(t)} y={H - padB + 12} textAnchor="middle">{PCT(t, t < 0.01 ? 1 : 0)}</text>
            <text className="axl" x={padL - 5} y={y(t) + 3.5} textAnchor="end">{PCT(t, t < 0.01 ? 1 : 0)}</text>
          </g>
        ))}
        <line x1={x(scale.lo)} y1={y(scale.lo)} x2={x(scale.hi)} y2={y(scale.hi)}
              stroke="var(--faint)" strokeDasharray="4 4" strokeWidth="1" />
        <path d={path} fill="none" stroke="var(--accent)" strokeWidth="1.6" strokeLinejoin="round" />
        {pts.map((p, i) => (
          <circle key={i} cx={x(p.predicted)} cy={y(p.observed)} r={2 + 3.4 * Math.sqrt(p.n / maxN)}
                  fill="var(--accent)" fillOpacity="0.85" stroke="var(--panel)" strokeWidth="1">
            <title>predicted {PCT(p.predicted, 2)} · observed {PCT(p.observed, 2)} · {fmtInt(p.n)} veteran-days</title>
          </circle>
        ))}
        <text className="axl" x={W - padR} y={H - 3} textAnchor="end">predicted →</text>
      </svg>
    </div>
  );
}

function Calibration({ report }: { report: ReportResponse }) {
  const scale = useMemo<LogScale>(() => {
    const vals = report.calibration.flatMap((b) => [b.predicted, b.observed]).filter((v) => v > 0);
    if (!vals.length) return { lo: 0.001, hi: 0.2 };
    return { lo: Math.min(...vals) * 0.7, hi: Math.max(...vals) * 1.4 };
  }, [report.calibration]);
  const worst = Object.entries(report.ece_by_need).sort((a, b) => b[1] - a[1])[0];

  if (!report.calibration.length) {
    return (
      <NotRun what="Reliability has no bins in this report"
              cmd="python -m leeward.eval.calibration"
              why="It is not a claim that the model is calibrated." />
    );
  }
  return (
    <div className="card">
      <div className="ch">
        <h3>Reliability: predicted risk against what happened</h3>
        <span className="sp" />
        <span className={worst && worst[1] > 0.03 ? "pillbadge pill-critical" : "pillbadge pill-healthy"}>
          worst need {worst ? `${NEED_LABEL[worst[0]] ?? worst[0]} ${NUM(worst[1], 4)}` : "—"} · bar is 0.03
        </span>
      </div>
      <p className="lede" style={{ marginBottom: 12 }}>
        Held-out window · equal-mass bins · simulated outcomes, synthetic cohort. Both axes
        are log and shared across the five panels; the dot area is how many veteran-days
        fell in that bin. On the dashed line the model said it and it happened.
      </p>
      <div className="rel">
        {NEEDS.filter((n) => report.calibration.some((b) => b.need === n)).map((n) => (
          <Reliability key={n} need={n} scale={scale}
                       bins={report.calibration.filter((b) => b.need === n)}
                       ece={report.ece_by_need[n]} />
        ))}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- #
// 3. Harm averted, against the three baselines
// --------------------------------------------------------------------------- #

const STRATEGY_ORDER = ["leeward", "rank_by_chronic", "rank_by_age", "random"];

function HarmAverted({ rows }: { rows: DecisionQualityRow[] }) {
  const ks = [...new Set(rows.map((r) => r.k))].sort((a, b) => a - b);
  const strategies = STRATEGY_ORDER.filter((s) => rows.some((r) => r.strategy === s));
  const value = (k: number, s: string) => rows.find((r) => r.k === k && r.strategy === s)?.harm_averted ?? 0;
  const max = Math.max(1e-9, ...rows.map((r) => r.harm_averted));

  const W = 560, H = 300, padL = 46, padR = 12, padT = 16, padB = 46;
  const plotH = H - padT - padB;
  const bandW = (W - padL - padR) / Math.max(1, ks.length);
  const barW = Math.min(30, (bandW * 0.72) / Math.max(1, strategies.length));
  // Round the top of the axis up to a 1-2-5 step, so the gridlines are numbers a person
  // would say out loud (0, 15, 30, 45, 60) rather than quarters of the tallest bar.
  const step = (() => {
    const raw = (max * 1.12) / 4;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    return [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  })();
  const top = step * 4;
  const y = (v: number) => padT + plotH * (1 - v / top);
  const gridlines = [0, 1, 2, 3, 4].map((i) => i * step);

  return (
    <svg className="chart2" viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label="Harm averted per day by strategy, at three daily capacities">
      {gridlines.map((g, i) => (
        <g key={i}>
          <line x1={padL} x2={W - padR} y1={y(g)} y2={y(g)} stroke="var(--line2)" strokeWidth="1" />
          <text className="axl" x={padL - 6} y={y(g) + 4} textAnchor="end">{g.toFixed(0)}</text>
        </g>
      ))}
      {ks.map((k, gi) => {
        const x0 = padL + gi * bandW + (bandW - barW * strategies.length) / 2;
        return (
          <g key={k}>
            {strategies.map((s, si) => {
              const v = value(k, s);
              const bx = x0 + si * barW;
              return (
                <g key={s}>
                  <rect x={bx} y={y(v)} width={barW - 4} height={Math.max(1, H - padB - y(v))}
                        rx="3" fill={STRATEGY_STEP[s] ?? "var(--faint)"}>
                    <title>{BASELINE_LABEL[s] ?? s} at {k} actions: {NUM(v)} severity-weighted events averted per day</title>
                  </rect>
                  <text className="axl" x={bx + (barW - 4) / 2} y={y(v) - 5} textAnchor="middle"
                        fill={s === "leeward" ? "var(--ink)" : "var(--muted)"}>{NUM(v, 1)}</text>
                </g>
              );
            })}
            <text className="axl" x={padL + gi * bandW + bandW / 2} y={H - padB + 17} textAnchor="middle">
              {k} actions/day
            </text>
          </g>
        );
      })}
      <line x1={padL} x2={W - padR} y1={H - padB} y2={H - padB} stroke="var(--line)" strokeWidth="1" />
      <text className="axl" x={padL - 6} y={padT - 4} textAnchor="end">events</text>
    </svg>
  );
}

function DecisionQuality({ report }: { report: ReportResponse }) {
  const rows = report.decision_quality;
  const ks = [...new Set(rows.map((r) => r.k))].sort((a, b) => a - b);
  const mid = ks.includes(40) ? 40 : ks[Math.floor(ks.length / 2)];
  const leeward = rows.find((r) => r.k === mid && r.strategy === "leeward")?.harm_averted ?? 0;
  const baselines = rows.filter((r) => r.k === mid && r.strategy !== "leeward");
  const best = baselines.sort((a, b) => b.harm_averted - a.harm_averted)[0];
  const lift = best && best.harm_averted > 0 ? leeward / best.harm_averted : null;

  if (!rows.length) {
    return (
      <NotRun what="Harm averted has no rows in this report"
              cmd="python -m leeward.eval.decision_quality"
              why="It is not a claim that the ranking beat anything." />
    );
  }
  return (
    <div className="card">
      <div className="ch">
        <h3>Harm averted per day, by who the care team calls</h3>
        <span className="sp" />
        {lift ? <span className="pillbadge pill-healthy">{NUM(lift)}× the best baseline at {mid}</span> : null}
      </div>
      <p className="lede" style={{ marginBottom: 6 }}>
        Every strategy gets the same K calls on the same held-out days, and is scored
        against the simulated outcomes. {EHA_REALIZED.line}
      </p>
      <HarmAverted rows={rows} />
      <div className="legend">
        {STRATEGY_ORDER.filter((s) => rows.some((r) => r.strategy === s)).map((s) => (
          <span key={s}><i style={{ background: STRATEGY_STEP[s], height: 9, width: 9, borderRadius: 3 }} />
            {BASELINE_LABEL[s] ?? s}</span>
        ))}
      </div>
      <p className="muted" style={{ fontSize: 12.5, marginBottom: 0 }}>
        Severity-weighted need-days averted, per day, at {mid} actions:{" "}
        <b>{NUM(leeward)}</b> against{" "}
        {baselines.map((b) => `${BASELINE_LABEL[b.strategy] ?? b.strategy} ${NUM(b.harm_averted)}`).join(", ")}.
      </p>
    </div>
  );
}

// --------------------------------------------------------------------------- #
// 4. Ablations and the fairness audit
// --------------------------------------------------------------------------- #

function Ablations({ report }: { report: ReportResponse }) {
  return (
    <div className="card">
      <div className="ch"><h3>Ablations: what each piece is worth</h3></div>
      {report.ablations.length ? (
        <>
          <p className="lede">Each row drops one block of the model and re-runs the same
            eval. A block that costs nothing to drop is a block we should not claim.</p>
          <div className="tablewrap auto">
            <table>
              <thead><tr><th>Dropped</th><th className="n">ECE</th><th className="n">Harm averted at 40</th></tr></thead>
              <tbody>
                {report.ablations.map((a) => (
                  <tr key={a.dropped}>
                    <td>{a.dropped}</td>
                    <td className="n">{NUM(a.ece, 4)}</td>
                    <td className="n">{NUM(a.harm_averted_at_40)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <NotRun what="Ablations have not been built yet"
                cmd="leeward/eval/ablate.py"
                why="It is not a claim that every block in the model earns its place." />
      )}
    </div>
  );
}

interface Verdict { tone: string; icon: "shield" | "alert"; headline: string; detail: string }

/**
 * The audit's own verdict, in the three states it actually has. "Not audited" is a
 * separate state from "passed" on purpose: an empty fairness table is the absence of
 * evidence, and must never be shown as the presence of a clean bill.
 */
/**
 * The pooled rates every ratio in the table was taken against, recovered from the rows.
 * Each stratum covers the same veteran-days, so summing across all of them scales
 * numerator and denominator alike and the rate holds.
 */
function pooled(rows: ReportResponse["fairness"]) {
  const events = rows.reduce((a, f) => a + f.n_events, 0);
  const reached = rows.reduce((a, f) => a + f.n_events * f.reach, 0);
  const measured = rows.every((f) => f.n_called !== null);
  const called = rows.reduce((a, f) => a + (f.n_called ?? 0), 0);
  const perStratum = new Set(rows.map((f) => f.stratum)).size || 1;
  return {
    events: Math.round(events / perStratum),
    fnr: events ? 1 - reached / events : 0,
    coverage: measured && events ? called / events : null,
  };
}

/**
 * Could *any* group have been flagged? At a pooled FNR of f, clearing the bar takes a group
 * FNR of f × 1.2, and an FNR above 1 does not exist. Above f = 0.833 the answer is no,
 * whatever the model does — which is the single most important thing to say next to a
 * report where nothing is flagged. `leeward/eval/fairness.py:flag_is_reachable`.
 */
const flagIsReachable = (fnr: number) => fnr * (1 + FNR_GAP) <= 1;

function auditVerdict(report: ReportResponse): Verdict {
  const flagged = report.fairness.filter((f) => f.flagged);
  const more = report.fairness.filter((f) => f.direction === "reached_more");
  if (!report.fairness.length) {
    return {
      tone: "warn", icon: "alert", headline: "Fairness audit not audited in this report",
      detail: "No groups were scored. This is the absence of an audit, not a pass. Run "
        + "`make report` against a scored cohort.",
    };
  }
  if (report.fairness_failed) {
    return {
      tone: "crit", icon: "alert",
      headline: `Fairness audit FAILED · ${flagged.length} of ${report.fairness.length} groups flagged`,
      detail: `These groups miss real events more than ${PCT(FNR_GAP, 0)} more often than the `
        + "cohort does. They are marked in the table below, in the order the audit returned "
        + "them. A failing audit is displayed, never suppressed.",
    };
  }
  const { fnr } = pooled(report.fairness);
  // "0 flagged" is only a pass if the bar could have been cleared. When it could not, the
  // banner says so in its headline rather than burying it in a footnote, because that is
  // the sentence that stops a reader concluding the model is fair from this screen.
  if (!flagIsReachable(fnr)) {
    return {
      tone: "warn", icon: "alert",
      headline: `Fairness audit · 0 of ${report.fairness.length} groups flagged, but no group could be`,
      detail: `The cohort misses ${PCT(fnr)} of its events, so flagging a group would take a `
        + `false-negative rate of ${NUM(fnr * (1 + FNR_GAP), 3)} — which is not a rate. This `
        + "is not a clean bill of health; it is a bar that cannot be crossed. Read the reach "
        + `and coverage columns instead, where ${more.length} group${more.length === 1 ? " is" : "s are"} `
        + "reached more than the cohort.",
    };
  }
  return {
    tone: "ok", icon: "shield",
    headline: `Fairness audit passed · 0 of ${report.fairness.length} groups flagged`,
    detail: `No group's false-negative rate is more than ${PCT(FNR_GAP, 0)} away from the `
      + `cohort's ${NUM(fnr, 3)}, and a group would have to reach ${NUM(fnr * (1 + FNR_GAP), 3)} `
      + "to be flagged. Every group audited is in the table below, flagged or not.",
  };
}

/**
 * A ratio against the cohort, centred on 1.00, with the ±20 % band it is judged by.
 *
 * The axis is log-scaled and runs 0.25×..4×. The FNR ratios all sit within a few points of
 * 1 and would be one flat line of dots on any axis; the reach and coverage ratios they are
 * shown beside span a factor of fifty, and clipping those to ±25 % would hide the entire
 * finding. A dot on the rail means the value is past the end of the axis, not at it.
 */
function RatioBar({ ratio, tone }: { ratio: number; tone: "crit" | "more" | "plain" }) {
  const lo = Math.log(0.25), hi = Math.log(4);
  const clamped = Math.max(0.25, Math.min(4, ratio || 0.25));
  const x = (v: number) => 5 + ((Math.log(v) - lo) / (hi - lo)) * 108;
  const ink = tone === "crit" ? "var(--crit)" : tone === "more" ? "var(--green)" : "var(--accent)";
  return (
    <svg className="ratiobar" viewBox="0 0 118 16" aria-hidden="true">
      <rect x={x(1 - FNR_GAP)} y="4" width={x(1 + FNR_GAP) - x(1 - FNR_GAP)} height="8" rx="2" fill="var(--panel2)" />
      {[1 - FNR_GAP, 1 + FNR_GAP].map((t) => (
        <line key={t} x1={x(t)} x2={x(t)} y1="2" y2="14" stroke="var(--amber)" strokeWidth="1" strokeDasharray="2 2" />
      ))}
      <line x1={x(1)} x2={x(1)} y1="1.5" y2="14.5" stroke="var(--faint)" strokeWidth="1" />
      <rect x={Math.min(x(1), x(clamped))} y="6" width={Math.max(1, Math.abs(x(clamped) - x(1)))} height="4" rx="2" fill={ink} />
      <circle cx={x(clamped)} cy="8" r="3.4" fill={ink} stroke="var(--panel)" strokeWidth="1" />
    </svg>
  );
}

/** A rate that may never have been measured. Null renders as a word, never as 0.0 %. */
const RATE = (v: number | null) => (v === null ? "not measured" : PCT(v, 2));
const TIMES = (v: number | null) => (v === null ? "—" : `${NUM(v)}×`);

const DIRECTION_LABEL: Record<string, string> = {
  reached_more: "reached more",
  reached_less: "reached less",
  on_par: "on par",
};

function Fairness({ report }: { report: ReportResponse }) {
  const v = auditVerdict(report);
  const { events, fnr: cohortFnr, coverage: cohortCoverage } = pooled(report.fairness);
  const unreachable = report.fairness.length > 0 && !flagIsReachable(cohortFnr);
  let previous = "";
  return (
    <div className="card">
      <div className="ch">
        <h3>Fairness audit: calibration, missed events and coverage by group</h3>
        <span className="sp" />
        <span className="pillbadge">simulated outcomes · synthetic cohort</span>
      </div>
      <div className={`auditbanner ${v.tone}`}>
        <span className={`ic ${v.tone === "crit" ? "ic-critical" : v.tone === "warn" ? "ic-medium" : "ic-low"}`}>
          {v.icon === "shield" ? <IconShield /> : <IconAlert />}
        </span>
        <span>
          <b>{v.headline}</b>
          <div className="muted" style={{ marginTop: 2 }}>{v.detail}</div>
        </span>
      </div>
      {/*
        The one line that has to be on screen. A judge who reads FNR 0.96 and nothing else
        concludes the model does not work; a judge who reads "0 flagged" and nothing else
        concludes it is fair. Both are wrong, and the arithmetic that makes them wrong is
        the same sentence, so it renders whenever there are rows — not only on a pass.
      */}
      {report.fairness.length > 0 && (
        <p className="lede">
          The false-negative rate is near 1 in every group because{" "}
          {report.model_rung === 0 ? "the model is prior-only at rung 0 and " : ""}
          the window holds <b>{fmtInt(events)} veteran-days with a need</b> and there are{" "}
          <b>{CALL_BUDGET} care-team calls a day</b> to spend on them.
          {cohortCoverage !== null && <> Those calls reached <b>{PCT(cohortCoverage, 2)}</b> of
            the events that happened.</>}{" "}
          {unreachable
            ? <>At a cohort rate of {NUM(cohortFnr, 3)}, being flagged would take a
              false-negative rate of {NUM(cohortFnr * (1 + FNR_GAP), 3)} — so the flag column
              cannot fire, and an empty one is not a result.</>
            : <>The cohort rate is {NUM(cohortFnr, 3)}.</>}{" "}
          <b>Read reach and coverage.</b> Reach asks whether anyone looked; coverage asks
          whether anyone phoned. They disagree, and where they disagree is where a care team
          would spend its next hire.
        </p>
      )}
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>Stratum</th>
              <th>Group</th>
              <th className="n">Events</th>
              <th className="n">ECE</th>
              <th className="n">FNR</th>
              <th className="n" title="The ratio the flag is computed from. It stays within a few points of 1 whatever the model does, which is why it is not the column to read.">FNR ×</th>
              <th className="n">Reach</th>
              <th style={{ width: 130 }}>Reach vs cohort</th>
              <th className="n">Ratio</th>
              <th className="n">Coverage @{CALL_BUDGET}</th>
              <th className="n">vs cohort</th>
              <th>Audit</th>
            </tr>
          </thead>
          <tbody>
            {report.fairness.map((f) => {
              const head = f.stratum !== previous;
              previous = f.stratum;
              const more = f.direction === "reached_more";
              return (
                <tr key={`${f.stratum}/${f.group}`}
                    className={f.flagged ? "flagrow" : more ? "morerow" : ""}>
                  <td className="muted">{head ? f.stratum.replace(/_/g, " ") : ""}</td>
                  <td>{f.group.replace(/_/g, " ")}</td>
                  <td className="n">{fmtInt(f.n_events)}</td>
                  <td className="n">{NUM(f.ece, 4)}</td>
                  <td className="n">{PCT(f.fnr)}</td>
                  <td className="n muted">{NUM(f.fnr_ratio_to_cohort)}×</td>
                  <td className="n">{PCT(f.reach, 2)}</td>
                  <td><RatioBar ratio={f.reach_ratio_to_cohort}
                                tone={f.flagged ? "crit" : more ? "more" : "plain"} /></td>
                  <td className="n">{NUM(f.reach_ratio_to_cohort)}×</td>
                  <td className="n">{RATE(f.coverage)}
                    {f.n_called !== null && <span className="muted"> ({fmtInt(f.n_called)})</span>}</td>
                  <td className="n">{TIMES(f.coverage_ratio_to_cohort)}</td>
                  <td>{f.flagged
                    ? <span className="rb rb-critical">flagged</span>
                    : more
                      ? <span className="rb rb-low">{DIRECTION_LABEL[f.direction]}</span>
                      : <span className="rb rb-quiet">{DIRECTION_LABEL[f.direction] ?? f.direction}</span>}</td>
                </tr>
              );
            })}
            {!report.fairness.length && (
              <tr><td colSpan={12} className="muted">
                No group was audited in this report — not audited is not the same as passed.
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="legend">
        <span>Shaded band: the ±{PCT(FNR_GAP, 0)} relative gap the audit tests, on a log axis</span>
        <span><i style={{ background: "var(--crit)", height: 9, width: 9, borderRadius: 9 }} /> flagged: misses more events than the cohort</span>
        <span><i style={{ background: "var(--green)", height: 9, width: 9, borderRadius: 9 }} /> reached more than the cohort — a result, not a pass</span>
        <span>Coverage @{CALL_BUDGET} counts the events that got one of the {CALL_BUDGET} daily calls; the figure in brackets is how many.</span>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- #

export function Report({ onSource }: { onSource: (s: Source) => void }) {
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getReport()
      .then((r) => { setReport(r); onSource(lastSource()); })
      .catch((e) => setError(String(e)));
  }, [onSource]);

  if (error) return <div className="page"><div className="empty"><div className="eh">Could not load the report</div>{error}</div></div>;
  if (!report) return <div className="page"><div className="skgrid">{[0, 1, 2, 3, 4].map((i) => <div key={i} className="sk" style={{ height: 96 }} />)}</div></div>;

  const missed = report.recovery.filter((r) => !r.covered).length;
  const eces = Object.entries(report.ece_by_need).sort((a, b) => b[1] - a[1]);
  const flagged = report.fairness.filter((f) => f.flagged).length;
  const audited = report.fairness.length;
  const ks = [...new Set(report.decision_quality.map((r) => r.k))].sort((a, b) => a - b);
  const mid = ks.includes(40) ? 40 : ks[Math.floor(ks.length / 2)];
  const leeward = report.decision_quality.find((r) => r.k === mid && r.strategy === "leeward")?.harm_averted;
  const bestBaseline = report.decision_quality
    .filter((r) => r.k === mid && r.strategy !== "leeward")
    .sort((a, b) => b.harm_averted - a.harm_averted)[0];
  const lift = leeward && bestBaseline?.harm_averted ? leeward / bestBaseline.harm_averted : null;

  // "0 of 33 flagged" is a pass only when a group could have been flagged. On this cohort
  // it cannot -- see `flagIsReachable` -- so the tile says which of the two it is showing
  // rather than colouring an unfalsifiable number green.
  const cohort = pooled(report.fairness);
  const barUnreachable = audited > 0 && !flagged && !flagIsReachable(cohort.fnr);
  const reachedMore = report.fairness.filter((f) => f.direction === "reached_more").length;

  let auditTile = "pillbadge pill-healthy";
  let auditWord = "passed";
  if (!audited) { auditTile = "pillbadge pill-watch"; auditWord = "not audited"; }
  else if (flagged) { auditTile = "pillbadge pill-critical"; auditWord = "failed"; }
  else if (barUnreachable) { auditTile = "pillbadge pill-watch"; auditWord = "bar unreachable"; }

  return (
    <div className="page dash">
      <div className="strip">
        <div className="stat">
          <div className="k">Which model produced this</div>
          <div className="vrow"><div className="v" style={{ fontSize: 19 }}>{RUNG_LABEL[report.model_rung] ?? `rung ${report.model_rung}`}</div></div>
          <div className="s">run {fmtRun(report.generated_at)}</div>
        </div>
        <div className="stat">
          <div className="k">Sampler</div>
          <div className="vrow">
            <div className="v">{report.rhat_max === null || report.rhat_max === undefined ? "—" : NUM(report.rhat_max, 3)}</div>
            <span className="muted" style={{ fontSize: 12 }}>max r-hat</span>
          </div>
          <div className="s">
            {report.rhat_max === null || report.rhat_max === undefined
              ? "prior-only: no MCMC ran, so there is no r-hat to report"
              : `${report.divergences ?? 0} divergences · bar is r-hat ≤ 1.01`}
          </div>
        </div>
        <div className="stat">
          <div className="k">{report.model_rung === 0 ? "Truth inside the prior interval" : "Truth inside its interval"}</div>
          <div className="vrow">
            <div className="v">{report.recovery.length - missed}/{report.recovery.length}</div>
            {report.recovery_coverage !== null && report.recovery_coverage !== undefined && (
              <span className={missed ? "pillbadge pill-watch" : "pillbadge pill-healthy"}>{PCT(report.recovery_coverage, 0)}</span>
            )}
          </div>
          <div className="s">bar is 0.90 · {missed} missed</div>
        </div>
        <div className="stat">
          <div className="k">Calibration, worst need</div>
          <div className="vrow">
            <div className="v">{eces.length ? NUM(eces[0][1], 4) : "—"}</div>
            <span className={eces.length && eces[0][1] > 0.03 ? "pillbadge pill-critical" : "pillbadge pill-healthy"}>bar 0.03</span>
          </div>
          <div className="s">{eces.length ? `${NEED_LABEL[eces[0][0]] ?? eces[0][0]} · mean ${NUM(eces.reduce((s, e) => s + e[1], 0) / eces.length, 4)}` : "not run"}</div>
        </div>
        <div className="stat hero">
          <div className="k">Harm averted at {mid ?? 40} actions/day</div>
          <div className="vrow">
            <div className="v">{leeward === undefined ? "—" : NUM(leeward)}</div>
            {lift ? <span className="delta up">▲ {NUM(lift)}×</span> : null}
          </div>
          <div className="s" title={EHA_REALIZED.line}>vs. the best baseline · {EHA_REALIZED.short}</div>
        </div>
        <div className={`stat ${flagged ? "crit" : ""}`}>
          <div className="k">Fairness audit</div>
          <div className="vrow">
            <div className="v">{flagged}/{audited}</div>
            <span className={auditTile}>{auditWord}</span>
          </div>
          <div className="s">
            {barUnreachable
              ? `no group can clear a ${PCT(FNR_GAP, 0)} gap at a cohort FNR of ${NUM(cohort.fnr, 2)} · ${reachedMore} reached more`
              : `groups flagged at a ${PCT(FNR_GAP, 0)} relative FNR gap`}
          </div>
        </div>
      </div>

      <h2>1 · Does the fit find the world it was trained on?</h2>
      <Recovery report={report} />

      <h2>2 · When it says 3%, does it happen 3% of the time?</h2>
      <Calibration report={report} />

      <h2>3 · Does the ranking avert more harm than the alternatives?</h2>
      <div className="row g2">
        <DecisionQuality report={report} />
        <Ablations report={report} />
      </div>

      <h2>4 · Do the misses fall evenly?</h2>
      <Fairness report={report} />

      <p className="muted" style={{ fontSize: 12.5 }}>
        Every number on this screen is <code>report/report.json</code> as <code>make report</code>{" "}
        wrote it, drawn without filtering. The same run also writes{" "}
        <code>report/recovery.csv</code>, <code>calibration.csv</code>,{" "}
        <code>decision_quality.csv</code> and <code>fairness.csv</code>, plus offline Plotly
        copies of these four charts. Outcomes are simulated and the cohort is synthetic.
      </p>
    </div>
  );
}

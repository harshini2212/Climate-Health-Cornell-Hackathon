/**
 * The proof. "How do you know it works?" gets a screen, not a footnote. Every section is
 * either a real number from `make report` or an honest "not run yet"; an empty fairness
 * table means "not audited", never "passed".
 */

import { useEffect, useState } from "react";
import { HBars } from "../components/Charts";
import { getReport, lastSource, type Source } from "../lib/api";
import { BASELINE_LABEL, NEED_LABEL, RUNG_LABEL } from "../lib/labels";
import type { ReportResponse } from "../lib/types";

function Pending({ what, cmd }: { what: string; cmd: string }) {
  return (
    <div className="pending"><span><b>{what}</b> has not been run yet · <code>{cmd}</code></span></div>
  );
}

export function Report({ onSource }: { onSource: (s: Source) => void }) {
  const [rep, setRep] = useState<ReportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getReport().then((r) => { setRep(r); onSource(lastSource()); }).catch((e) => setError(String(e)));
  }, [onSource]);

  if (error) return <div className="page"><div className="empty"><div className="eh">Report unavailable</div>{error}</div></div>;
  if (!rep) return <div className="page"><div className="skgrid"><div className="sk" style={{ height: 96 }} /><div className="sk" style={{ height: 96 }} /><div className="sk" style={{ height: 96 }} /><div className="sk" style={{ height: 96 }} /><div className="sk" style={{ height: 96 }} /></div></div>;

  const ks = [...new Set(rep.decision_quality.map((r) => r.k))].sort((a, b) => a - b);
  const at = (k: number, s: string) => rep.decision_quality.find((r) => r.k === k && r.strategy === s)?.harm_averted ?? 0;
  const k40 = ks.includes(40) ? 40 : ks[0];
  const ratio = k40 !== undefined && at(k40, "rank_by_age") > 0 ? at(k40, "leeward") / at(k40, "rank_by_age") : null;
  const fairnessRows = rep.fairness;

  return (
    <div className="page dash">
      <div className="strip">
        <div className="stat">
          <div className="k">Model</div>
          <div className="vrow"><div className="v" style={{ fontSize: 18 }}>{RUNG_LABEL[rep.model_rung]}</div></div>
          <div className="s">say the rung on stage</div>
        </div>
        <div className="stat">
          <div className="k">Worst r-hat</div>
          <div className="vrow"><div className="v">{rep.rhat_max == null ? "—" : rep.rhat_max.toFixed(3)}</div></div>
          <div className="s">{rep.rhat_max == null ? "no MCMC at rung 0" : rep.rhat_max < 1.05 ? "converged" : "did not converge"}</div>
        </div>
        <div className="stat">
          <div className="k">Divergences</div>
          <div className="vrow"><div className="v">{rep.divergences == null ? "—" : rep.divergences}</div></div>
          <div className="s">after warmup</div>
        </div>
        <div className="stat hero">
          <div className="k">Harm averted vs oldest-first</div>
          <div className="vrow"><div className="v">{ratio ? `${ratio.toFixed(1)}×` : "—"}</div></div>
          <div className="s">realised, at {k40 ?? 40} calls a day</div>
        </div>
        <div className="stat">
          <div className="k">Recovery coverage</div>
          <div className="vrow"><div className="v">{rep.recovery_coverage == null ? "—" : `${(rep.recovery_coverage * 100).toFixed(0)}%`}</div></div>
          <div className="s">true coefficients inside the 90% interval</div>
        </div>
      </div>

      <div className="card">
        <div className="ch">
          <h3>Decision quality</h3>
          <span className="sub">realised harm averted per day on the held-out window, from the planted outcomes</span>
          <span className="sp" />
          {rep.generated_at && <span className="tag">generated {rep.generated_at.replace("T", " ")}</span>}
        </div>
        {ks.length === 0 ? (
          <Pending what="Decision quality" cmd="python -m leeward.eval.decision_quality" />
        ) : (
          <div className="row g3">
            {ks.map((k) => {
              const rows = ["leeward", "rank_by_age", "rank_by_chronic", "random"].map((s) => ({ label: BASELINE_LABEL[s], value: at(k, s), lead: s === "leeward" }));
              const r = at(k, "rank_by_age") > 0 ? at(k, "leeward") / at(k, "rank_by_age") : null;
              return (
                <div key={k}>
                  <div className="kx"><b>{k} calls a day</b>{r && <span className="ratio">{r.toFixed(1)}× oldest-first</span>}</div>
                  <HBars rows={rows} max={Math.max(...ks.map((kk) => at(kk, "leeward")))} />
                </div>
              );
            })}
          </div>
        )}
        <div className="muted small" style={{ marginTop: 12 }}>
          Same K-call budget for every strategy. The action list's own "expected" ratio is smaller because it counts expected harm on the day; this counts what the simulated outcomes then showed. Both are correct and they are labelled.
        </div>
      </div>

      <div className="row g2">
        <div className="card">
          <div className="ch"><h3>Calibration</h3><span className="sub">predicted 30% risks should happen about 30% of the time</span></div>
          {rep.calibration.length === 0 ? (
            <Pending what="Calibration" cmd="python -m leeward.eval.calibration" />
          ) : (
            <table>
              <thead><tr><th>Need</th><th className="n">Predicted</th><th className="n">Observed</th><th className="n">n</th></tr></thead>
              <tbody>
                {rep.calibration.map((c, i) => (
                  <tr key={i}><td>{NEED_LABEL[c.need] ?? c.need}</td><td className="n">{(c.predicted * 100).toFixed(1)}%</td><td className="n">{(c.observed * 100).toFixed(1)}%</td><td className="n">{c.n}</td></tr>
                ))}
              </tbody>
            </table>
          )}
          {Object.keys(rep.ece_by_need).length > 0 && (
            <div className="chips" style={{ marginTop: 10 }}>
              {Object.entries(rep.ece_by_need).map(([need, ece]) => <span key={need} className={`flagpill${ece < 0.03 ? " info" : " crit"}`}>{NEED_LABEL[need] ?? need} · ECE {ece.toFixed(3)}</span>)}
            </div>
          )}
        </div>

        <div className="card">
          <div className="ch"><h3>Parameter recovery</h3><span className="sub">planted truth inside the posterior's 90% interval</span></div>
          {rep.recovery.length === 0 ? (
            <Pending what="Recovery" cmd="python -m leeward.eval.recovery" />
          ) : (
            <table>
              <thead><tr><th>Parameter</th><th className="n">Truth</th><th className="n">Posterior</th><th className="n">90% interval</th><th></th></tr></thead>
              <tbody>
                {rep.recovery.map((r) => (
                  <tr key={r.parameter}><td>{r.parameter}</td><td className="n">{r.truth.toFixed(2)}</td><td className="n">{r.post_mean.toFixed(2)}</td><td className="n">{r.lo90.toFixed(2)} – {r.hi90.toFixed(2)}</td><td>{r.covered ? <span className="rb rb-low">covered</span> : <span className="rb rb-critical">missed</span>}</td></tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="card">
        <div className="ch">
          <h3>Fairness audit</h3>
          <span className="sub">calibration and false-negative rate by borough, HVI band, evacuation zone, income, caregiver status, race and ethnicity</span>
          <span className="sp" />
          {fairnessRows.length > 0 && (rep.fairness_failed ? <span className="rb rb-critical">gaps flagged</span> : <span className="rb rb-low">no gap over 20%</span>)}
        </div>
        {fairnessRows.length === 0 ? (
          <Pending what="Fairness audit" cmd="python -m leeward.eval.fairness" />
        ) : (
          <table>
            <thead><tr><th>Stratum</th><th>Group</th><th className="n">n</th><th className="n">ECE</th><th className="n">FNR</th><th className="n">vs cohort</th><th></th></tr></thead>
            <tbody>
              {fairnessRows.map((f, i) => (
                <tr key={i}><td>{f.stratum}</td><td>{f.group}</td><td className="n">{f.n}</td><td className="n">{f.ece.toFixed(3)}</td><td className="n">{(f.fnr * 100).toFixed(1)}%</td><td className="n">{f.fnr_ratio_to_cohort.toFixed(2)}×</td><td>{f.flagged ? <span className="rb rb-critical">flagged</span> : <span className="rb rb-quiet">ok</span>}</td></tr>
              ))}
            </tbody>
          </table>
        )}
        <div className="muted small" style={{ marginTop: 10 }}>A failing audit is displayed, never suppressed. When a group's false-negative rate runs more than 20% above the cohort, the allocator can apply a per-group capacity floor.</div>
      </div>

      {rep.ablations.length > 0 && (
        <div className="card">
          <div className="ch"><h3>Ablations</h3><span className="sub">what each model component buys</span></div>
          <table>
            <thead><tr><th>Dropped</th><th className="n">ECE</th><th className="n">Harm averted at 40</th></tr></thead>
            <tbody>{rep.ablations.map((a) => <tr key={a.dropped}><td>{a.dropped}</td><td className="n">{a.ece.toFixed(3)}</td><td className="n">{a.harm_averted_at_40.toFixed(1)}</td></tr>)}</tbody>
          </table>
        </div>
      )}
    </div>
  );
}

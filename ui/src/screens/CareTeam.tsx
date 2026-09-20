import { useCallback, useEffect, useState } from "react";
import { HBars } from "../components/Charts";
import { CapacitySlider } from "../components/CapacitySlider";
import { HarmCounter } from "../components/HarmCounter";
import { IconArrow, IconDownload } from "../components/Icons";
import { exportUrl, lastSource, postActions, type Source } from "../lib/api";
import { ACTION_LABEL, BASELINE_LABEL, TIER_BADGE, TIER_LABEL, fmtDate, fmtInt } from "../lib/labels";
import { DEFAULT_CAPACITY, TIERS, type ActionsResponse } from "../lib/types";
import type { VeteranFocus } from "./Week";

interface Props {
  scenario: string;
  /** Set when the week board hands over a day; null means "whatever the API calls today". */
  date?: string | null;
  onSource: (s: Source) => void;
  onOpenVeteran: (f: VeteranFocus) => void;
}

export function CareTeam({ scenario, date, onSource, onOpenVeteran }: Props) {
  const [calls, setCalls] = useState(DEFAULT_CAPACITY.call);
  const [resp, setResp] = useState<ActionsResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [roundTrip, setRoundTrip] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tierFilter, setTierFilter] = useState<string>("all");
  const [source, setSource] = useState<Source>("fixture");

  const load = useCallback(
    async (n: number) => {
      setBusy(true);
      setError(null);
      const t0 = performance.now();
      try {
        const r = await postActions({ date: date ?? "", capacity: { ...DEFAULT_CAPACITY, call: n }, scenario });
        setResp(r);
        setRoundTrip(Math.round(performance.now() - t0));
        setSource(lastSource());
        onSource(lastSource());
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [scenario, date, onSource],
  );

  useEffect(() => {
    void load(calls);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load]);

  const commit = (n: number) => {
    setCalls(n);
    void load(n);
  };

  if (error) return <div className="page"><div className="empty"><div className="eh">Could not load the action list</div>{error}</div></div>;
  if (!resp) return <div className="page dash"><div className="skgrid">{Array.from({ length: 5 }, (_, i) => <div key={i} className="sk" style={{ height: 96 }} />)}</div><div className="sk" style={{ height: 420 }} /></div>;

  const callsUsed = resp.actions.filter((a) => a.capacity_bucket === "call").length;
  const rows = tierFilter === "all" ? resp.actions : resp.actions.filter((a) => a.tier === tierFilter);
  const human = resp.actions.filter((a) => a.capacity_bucket !== "free").length;
  const age = resp.baselines.find((b) => b.name === "rank_by_age")?.total_eha;
  const ratio = age ? resp.total_eha / age : null;
  const barRows = [{ name: "leeward", total_eha: resp.total_eha }, ...resp.baselines.filter((b) => b.name !== "leeward")].map((b) => ({ label: BASELINE_LABEL[b.name] ?? b.name, value: b.total_eha, lead: b.name === "leeward" }));

  return (
    <div className="page dash">
      <div className="strip">
        <div className="stat hero">
          <div className="k">Expected harm averted today</div>
          <div className="vrow"><div className="v"><HarmCounter total={resp.total_eha} /></div>{ratio && <span className="delta up">▲ {ratio.toFixed(2)}×</span>}</div>
          <div className="s">severity-weighted need-days · vs. oldest-first</div>
        </div>
        <div className="stat">
          <div className="k">Calls used</div>
          <div className="vrow"><div className="v">{callsUsed}<small>/ {calls}</small></div></div>
          <div className="prog" style={{ marginTop: 10 }}><i style={{ width: `${(100 * callsUsed) / calls}%` }} /></div>
        </div>
        {TIERS.filter((t) => t !== "everyday").map((t) => (
          <div className={`stat${t === "act_now" ? " crit" : ""}`} key={t}>
            <div className="k">{TIER_LABEL[t]}</div>
            <div className="vrow"><div className="v">{fmtInt(resp.counts_by_tier[t] ?? 0)}</div></div>
            <div className="s">{t === "act_now" ? "call today, up to 3 actions" : t === "find_out" ? "wide interval: 3-minute check-in" : "verified text, self-directed"}</div>
          </div>
        ))}
      </div>

      <div className="row g2">
        <div className="card">
          <div className="ch"><h3>Capacity</h3><span className="sub">drag, release, the list re-ranks</span><span className="sp" />{roundTrip !== null && <span className="tag">{roundTrip} ms</span>}</div>
          <CapacitySlider value={calls} onCommit={commit} disabled={busy} />
        </div>
        <div className="card">
          <div className="ch"><h3>Versus the baselines</h3><span className="sub">same capacity, different ranking · expected</span></div>
          <HBars rows={barRows} />
        </div>
      </div>

      <div>
        <div className="tbar">
          <span className="ct"><b>{fmtDate(resp.date)}</b> · {fmtInt(resp.n_panel)} on the panel · {fmtInt(human)} people to reach by hand · {fmtInt(resp.n_selected - human)} automated texts</span>
          <span className="sp" />
          <div className="ctrls">
            <span className={`iv${tierFilter === "all" ? " on" : ""}`} onClick={() => setTierFilter("all")}>All</span>
            {TIERS.map((t) => (
              <span key={t} className={`iv${tierFilter === t ? " on" : ""}`} onClick={() => setTierFilter(t)}>{TIER_LABEL[t]}</span>
            ))}
          </div>
          {source === "api" ? (
            <a className="pill" href={exportUrl(resp.date)} download title="Partner sheet CSV with consent flags; rows without consent are excluded, never redacted"><IconDownload /> Partner sheet</a>
          ) : (
            <span className="pill" title="The partner sheet is served by the API; start it to export" style={{ opacity: .55 }}><IconDownload /> Partner sheet</span>
          )}
        </div>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th className="n">#</th>
                <th>Veteran</th>
                <th>Tier</th>
                <th>Action</th>
                <th>Top driver</th>
                <th className="n">Expected harm averted</th>
                <th>Owner</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 400).map((a) => (
                <tr key={a.action_id} className="click" onClick={() => onOpenVeteran({ veteranId: a.veteran_id, actionId: a.action_id, date: resp.date })}>
                  <td className="n">{a.rank}</td>
                  <td>{a.name_display}<span className="sub">{a.borough} · {a.modzcta}</span></td>
                  <td><span className={TIER_BADGE[a.tier]}>{TIER_LABEL[a.tier]}</span></td>
                  <td className="wrap">{ACTION_LABEL[a.action] ?? a.action}<span className="sub">{a.rationale}</span></td>
                  <td>{a.top_driver ? <span className="tag">{a.top_driver}</span> : <span className="muted">—</span>}</td>
                  <td className="n">{a.eha.toFixed(2)}</td>
                  <td className="muted">{a.owner.replace("_", " ")}</td>
                  <td><IconArrow /></td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length > 400 && <div className="muted small" style={{ padding: "10px 12px" }}>Showing the first 400 of {fmtInt(rows.length)} rows.</div>}
        </div>
      </div>
    </div>
  );
}

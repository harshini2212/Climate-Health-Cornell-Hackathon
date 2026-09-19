import { useCallback, useEffect, useState } from "react";
import { CapacitySlider } from "../components/CapacitySlider";
import { HarmCounter } from "../components/HarmCounter";
import { lastSource, postActions, type Source } from "../lib/api";
import { ACTION_LABEL, TIER_BADGE, TIER_LABEL } from "../lib/labels";
import { DEFAULT_CAPACITY, TIERS, type ActionsResponse } from "../lib/types";

export function CareTeam({ scenario, onSource }: { scenario: string; onSource: (s: Source) => void }) {
  const [calls, setCalls] = useState(DEFAULT_CAPACITY.call);
  const [resp, setResp] = useState<ActionsResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [roundTrip, setRoundTrip] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tierFilter, setTierFilter] = useState<string>("all");

  const load = useCallback(
    async (n: number) => {
      setBusy(true);
      setError(null);
      const t0 = performance.now();
      try {
        const r = await postActions({ date: "", capacity: { ...DEFAULT_CAPACITY, call: n }, scenario });
        setResp(r);
        setRoundTrip(Math.round(performance.now() - t0));
        onSource(lastSource());
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [scenario, onSource],
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
  if (!resp) return <div className="page"><div className="skgrid"><div className="sk" style={{ height: 96 }} /><div className="sk" style={{ height: 96 }} /><div className="sk" style={{ height: 96 }} /></div></div>;

  const callsUsed = resp.actions.filter((a) => a.capacity_bucket === "call").length;
  const rows = tierFilter === "all" ? resp.actions : resp.actions.filter((a) => a.tier === tierFilter);

  return (
    <div className="page dash">
      <div className="strip">
        <div className="stat hero">
          <div className="k">Expected harm averted today</div>
          <div className="vrow"><div className="v"><HarmCounter total={resp.total_eha} /></div></div>
          <div className="s">severity-weighted need-days · {resp.n_selected} actions</div>
        </div>
        <div className="stat">
          <div className="k">Calls used</div>
          <div className="vrow"><div className="v">{callsUsed} / {calls}</div></div>
          <div className="prog" style={{ marginTop: 8 }}><i style={{ width: `${(100 * callsUsed) / calls}%` }} /></div>
        </div>
        {TIERS.filter((t) => t !== "everyday").map((t) => (
          <div className={`stat${t === "act_now" ? " crit" : ""}`} key={t}>
            <div className="k">{TIER_LABEL[t]}</div>
            <div className="vrow"><div className="v">{resp.counts_by_tier[t] ?? 0}</div></div>
            <div className="s">{t === "act_now" ? "call today, up to 3 actions" : t === "find_out" ? "wide interval: 3-minute check-in" : t === "self_serve" ? "verified text, self-directed" : "monthly wellness plan"}</div>
          </div>
        ))}
      </div>

      <div className="row g2">
        <div className="card">
          <div className="ch"><h3>Capacity</h3><span className="muted" style={{ fontSize: 12 }}>drag, release, the list re-ranks</span><span className="sp" /><span className="pillbadge">{roundTrip} ms</span></div>
          <CapacitySlider value={calls} onCommit={commit} disabled={busy} />
        </div>
        <div className="card">
          <div className="ch"><h3>Versus the baselines</h3><span className="sp" /><span className="muted" style={{ fontSize: 12 }}>same capacity, different ranking</span></div>
          <div className="bars">
            {[{ name: "leeward", total_eha: resp.total_eha }, ...resp.baselines.filter((b) => b.name !== "leeward")].map((b) => {
              const max = Math.max(resp.total_eha, ...resp.baselines.map((x) => x.total_eha), 1e-9);
              const label: Record<string, string> = { leeward: "Leeward", rank_by_age: "Oldest first", rank_by_chronic: "Most conditions first", random: "Random" };
              return (
                <div className={`barrow${b.name === "leeward" ? " lead" : ""}`} key={b.name}>
                  <span className="lb">{label[b.name] ?? b.name}</span>
                  <div className="bt"><div className="bf" style={{ width: `${(100 * b.total_eha) / max}%` }} /></div>
                  <span className="bv">{b.total_eha.toFixed(1)}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div>
        <div className="tbar">
          <span className="ct"><b>{resp.date}</b> · {resp.n_panel.toLocaleString()} on the panel · cut at {callsUsed} calls</span>
          <span className="sp" />
          <div className="ctrls" style={{ margin: 0 }}>
            <span className={`iv${tierFilter === "all" ? " on" : ""}`} onClick={() => setTierFilter("all")}>All</span>
            {TIERS.map((t) => (
              <span key={t} className={`iv${tierFilter === t ? " on" : ""}`} onClick={() => setTierFilter(t)}>{TIER_LABEL[t]}</span>
            ))}
          </div>
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
                <th className="n">EHA</th>
                <th>Owner</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((a) => (
                <tr key={a.action_id}>
                  <td className="n">{a.rank}</td>
                  <td>{a.name_display}<div className="muted" style={{ fontSize: 11.5 }}>{a.borough} · {a.modzcta}</div></td>
                  <td><span className={TIER_BADGE[a.tier]}>{TIER_LABEL[a.tier]}</span></td>
                  <td className="wrap">{ACTION_LABEL[a.action] ?? a.action}<div className="muted" style={{ fontSize: 12 }}>{a.rationale}</div></td>
                  <td>{a.top_driver ? <span className="tag">{a.top_driver}</span> : <span className="muted">—</span>}</td>
                  <td className="n">{a.eha.toFixed(2)}</td>
                  <td className="muted">{a.owner.replace("_", " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

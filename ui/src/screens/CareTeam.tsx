import { useCallback, useEffect, useState } from "react";
import { CapacitySlider } from "../components/CapacitySlider";
import { HarmCounter } from "../components/HarmCounter";
import { TierBadge } from "../components/TierBadge";
import { lastSource, postActions, type Source } from "../lib/api";
import { TIER_LABEL, TIER_STEP } from "../lib/colors";
import { DEFAULT_CAPACITY, TIERS, type ActionsResponse } from "../lib/types";

const ACTION_LABEL: Record<string, string> = {
  care_team_call: "Care-team call",
  check_in_call: "3-minute check-in",
  backup_power_plan: "Backup-power plan",
  cold_chain_plan: "Cold-chain plan",
  early_refill: "Early refill",
  switch_to_local_pickup: "Switch to local pickup",
  cooling_center_ride: "Cooling-center ride",
  clean_air_room: "Clean-air room",
  alt_site_booking: "Alternate-site booking",
  evacuation_assist: "Evacuation assist",
  assign_buddy: "Assign buddy",
  pharmacist_med_review: "Pharmacist review",
  controlled_substance_bridge: "Controlled-substance bridge",
  heap_application: "HEAP application",
  verified_text: "Verified text",
};

interface Props {
  scenario: string;
  /** Set when the week board hands over a day; null means "whatever the API calls today". */
  date?: string | null;
  onSource: (s: Source) => void;
}

export function CareTeam({ scenario, date, onSource }: Props) {
  const [calls, setCalls] = useState(DEFAULT_CAPACITY.call);
  const [resp, setResp] = useState<ActionsResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [roundTrip, setRoundTrip] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (n: number) => {
      setBusy(true);
      setError(null);
      const t0 = performance.now();
      try {
        const r = await postActions({
          date: date ?? resp?.date ?? "",
          capacity: { ...DEFAULT_CAPACITY, call: n },
          scenario,
        });
        setResp(r);
        setRoundTrip(Math.round(performance.now() - t0));
        onSource(lastSource());
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    // resp?.date only matters after the first load; with no date from the week board the
    // first request sends "" and the fixture/API resolve the demo date themselves.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [scenario, date],
  );

  useEffect(() => {
    void load(calls);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load]);

  const commit = (n: number) => {
    setCalls(n);
    void load(n);
  };

  const cutAt = resp?.actions.filter((a) => a.capacity_bucket === "call").length ?? 0;

  return (
    <div className="screen careteam">
      <aside className="panel">
        <h2>Today's action list</h2>
        <CapacitySlider value={calls} onCommit={commit} disabled={busy} />
        {resp && <HarmCounter total={resp.total_eha} baselines={resp.baselines} />}
        {resp && (
          <div>
            <h3>By tier</h3>
            <div className="tiers">
              {TIERS.map((t) => (
                <span key={t} className="tier">
                  <span className="dot" style={{ background: TIER_STEP[t] }} />
                  {TIER_LABEL[t]} · {resp.counts_by_tier[t] ?? 0}
                </span>
              ))}
            </div>
          </div>
        )}
        {resp && (
          <div style={{ fontSize: 12, color: "var(--muted)" }}>
            {resp.n_selected} actions for {resp.n_panel.toLocaleString()} veterans on the panel.
            Model rung {resp.model_rung}. Last request {roundTrip} ms.
          </div>
        )}
      </aside>

      <section className="table-wrap">
        {error && <div className="empty">{error}</div>}
        {!resp && !error && <div className="empty">Loading the action list…</div>}
        {resp && (
          <>
            <div className="meta">
              <span>
                <strong>{resp.date}</strong>
              </span>
              <span>
                Cut at <strong>{cutAt} calls</strong> of {calls} available
              </span>
              <span>
                Total EHA <strong>{resp.total_eha.toFixed(1)}</strong>
              </span>
            </div>
            <table>
              <thead>
                <tr>
                  <th className="num">#</th>
                  <th>Veteran</th>
                  <th>Tier</th>
                  <th>Action</th>
                  <th>Top driver</th>
                  <th className="num">EHA</th>
                  <th>Owner</th>
                </tr>
              </thead>
              <tbody>
                {resp.actions.map((a) => (
                  <tr key={a.action_id}>
                    <td className="num">{a.rank}</td>
                    <td>
                      {a.name_display}
                      <div className="sub">
                        {a.borough} · {a.modzcta}
                      </div>
                    </td>
                    <td>
                      <TierBadge tier={a.tier} />
                    </td>
                    <td>
                      {ACTION_LABEL[a.action] ?? a.action}
                      <div className="sub">{a.rationale}</div>
                    </td>
                    <td>{a.top_driver ? <span className="chip">{a.top_driver}</span> : <span className="sub">—</span>}</td>
                    <td className="num">{a.eha.toFixed(2)}</td>
                    <td>{a.owner.replace("_", " ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </section>
    </div>
  );
}

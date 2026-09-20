import { useEffect, useMemo, useState } from "react";
import type { ScreenKey } from "../App";
import { Spark } from "../components/Charts";
import { IconAlert, IconArrow, IconBolt, IconBuilding, IconDrop, IconMail, IconSun } from "../components/Icons";
import { getForecast, lastSource, postActions, type Source } from "../lib/api";
import { ACTION_LABEL, EHA_EXPECTED, TIER_BADGE, TIER_LABEL, fmtDate, fmtInt, handleFor } from "../lib/labels";
import { DEFAULT_CAPACITY, type ActionsResponse, type ForecastResponse } from "../lib/types";

interface DayRow {
  date: string;
  heat: number;
  pm25: number;
  heatAlert: boolean;
  flood: number;
  outage: number;
  mail: number;
}

function summarise(f: ForecastResponse): DayRow[] {
  return f.dates.map((date) => {
    const rows = f.zips.filter((z) => z.date === date);
    const n = rows.length || 1;
    return {
      date,
      heat: rows.reduce((s, z) => s + z.heat_index_max_f, 0) / n,
      pm25: rows.reduce((m, z) => Math.max(m, z.pm25), 0),
      heatAlert: rows.some((z) => z.heat_alert),
      flood: rows.filter((z) => z.flood_warning || z.flash_flood_emergency).length,
      outage: rows.filter((z) => z.outage_frac >= 0.2).length,
      mail: rows.filter((z) => z.mail_delivery_disrupted).length,
    };
  });
}

export function Forecast({ scenario, day, onSource, onOpen }: { scenario: string; day?: number; onSource: (s: Source) => void; onOpen: (s: ScreenKey) => void }) {
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [actions, setActions] = useState<ActionsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getForecast(scenario, day)
      .then((f) => {
        setForecast(f);
        onSource(lastSource());
        return postActions({ date: f.dates[0], capacity: { ...DEFAULT_CAPACITY }, scenario });
      })
      .then(setActions)
      .catch((e) => setError(String(e)));
  }, [scenario, day, onSource]);

  const days = useMemo(() => (forecast ? summarise(forecast) : []), [forecast]);
  const down = forecast?.facilities.filter((f) => f.site_down) ?? [];
  const landfall = days.find((d) => d.flood > 0);
  const firstHeat = days.find((d) => d.heatAlert);
  const alertDays = days.filter((d) => d.heatAlert || d.flood > 0).length;
  const callsUsed = actions?.actions.filter((a) => a.capacity_bucket === "call").length ?? 0;
  const callsCap = actions?.capacity.call ?? DEFAULT_CAPACITY.call;
  const leeward = actions?.total_eha ?? 0;
  const age = actions?.baselines.find((b) => b.name === "rank_by_age")?.total_eha;
  const ratio = age ? leeward / age : null;
  const peakMail = Math.max(0, ...days.map((d) => d.mail));
  const peakOutage = Math.max(0, ...days.map((d) => d.outage));

  if (error) return <div className="page"><div className="empty"><div className="eh">Could not load</div>{error}</div></div>;
  if (!forecast || !actions) return <div className="page dash"><div className="skgrid">{Array.from({ length: 5 }, (_, i) => <div key={i} className="sk" style={{ height: 96 }} />)}</div><div className="row g2"><div className="sk" style={{ height: 420 }} /><div className="sk" style={{ height: 420 }} /></div></div>;

  const brief = [
    `It is ${fmtDate(forecast.dates[0])}. ${fmtInt(actions.n_panel)} veterans are on the panel.`,
    landfall ? `A coastal flood warning covers ${landfall.flood} ZIPs on ${fmtDate(landfall.date)}, with ${peakOutage} ZIPs facing an outage and mail delivery disrupted in ${peakMail}.` : "No flood warning in the window.",
    down.length ? `${down.map((d) => d.name).join(", ")} is down: dialysis, infusion and OTP patients there need an alternate site booked before the closure bites.` : "All 14 VA sites are open.",
    firstHeat ? `A heat alert starts ${fmtDate(firstHeat.date)}, while power is still being restored.` : "No heat alert in the window.",
    `At ${callsCap} calls today the list is expected to avert ${leeward.toFixed(0)} severity-weighted need-days${ratio ? `, ${ratio.toFixed(2)}× calling the oldest patients first` : ""}.`,
  ].join(" ");

  return (
    <div className="page dash">
      <div className="strip">
        <div className="stat">
          <div className="k">Veterans on the panel</div>
          <div className="vrow"><div className="v">{fmtInt(actions.n_panel)}</div></div>
          <div className="s">rates from ACS, PLACES and emPOWER</div>
        </div>
        <div className="stat crit">
          <div className="k">Act now today</div>
          <div className="vrow"><div className="v">{actions.counts_by_tier.act_now ?? 0}</div></div>
          <div className="s">narrow interval, high risk, or site-dependent</div>
        </div>
        <div className="stat">
          <div className="k">Calls the team can make</div>
          <div className="vrow"><div className="v">{callsUsed}<small>/ {callsCap}</small></div></div>
          <div className="prog" style={{ marginTop: 10 }}><i style={{ width: `${(100 * callsUsed) / callsCap}%` }} /></div>
        </div>
        <div className="stat hero">
          <div className="k">Expected harm averted today</div>
          <div className="vrow"><div className="v">{leeward.toFixed(1)}</div>{ratio && <span className="delta up">▲ {ratio.toFixed(2)}×</span>}</div>
          <div className="s" title={EHA_EXPECTED.line}>vs. oldest-first, same {callsCap} calls · {EHA_EXPECTED.short}</div>
        </div>
        <div className="stat">
          <div className="k">VA sites down</div>
          <div className="vrow"><div className="v">{down.length}</div>{down.length ? <span className="pillbadge pill-critical">station {down.map((d) => d.facility_id).join(", ")}</span> : <span className="pillbadge pill-healthy">all open</span>}</div>
          <div className="s">of 14 · {alertDays} of {days.length} days under an alert</div>
        </div>
      </div>

      <div className="row g2">
        <div className="card">
          <div className="ch"><h3>Seven-day hazard</h3><span className="sub">citywide · NWS + AirNow</span><span className="sp" /><span className="tag">{forecast.scenario}</span></div>
          <div className="row gh">
            <div>
              <div className="chart-title">Max heat index, mean over ZIPs · hinge at 82 °F (NYC Health)</div>
              <Spark dates={days.map((d) => d.date)} values={days.map((d) => d.heat)} threshold={82} unit="°F" />
            </div>
            <div>
              <div className="chart-title">PM2.5, worst ZIP · smoke alert at 55.5 µg/m³</div>
              <Spark dates={days.map((d) => d.date)} values={days.map((d) => d.pm25)} threshold={55.5} />
            </div>
          </div>
          <table style={{ marginTop: 14 }}>
            <thead><tr><th>Day</th><th className="n">Heat idx</th><th className="n">PM2.5</th><th className="n">Flood ZIPs</th><th className="n">Outage ZIPs</th><th className="n">Mail disrupted</th><th>Flags</th></tr></thead>
            <tbody>
              {days.map((d) => (
                <tr key={d.date}>
                  <td>{fmtDate(d.date)}</td>
                  <td className="n">{d.heat.toFixed(0)}</td>
                  <td className="n">{d.pm25.toFixed(0)}</td>
                  <td className="n">{d.flood || "—"}</td>
                  <td className="n">{d.outage || "—"}</td>
                  <td className="n">{d.mail || "—"}</td>
                  <td>
                    {d.flood > 0 && <span className="flagpill crit" style={{ marginRight: 4 }}>flood warning</span>}
                    {d.heatAlert && <span className="flagpill">heat alert</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="stack">
          <div className="card">
            <div className="ch"><h3>Needs attention</h3><span className="sp" /><span className="aipill">◆ from the forecast</span></div>
            <div className="insights one">
              {down.map((d) => (
                <div className="insight flat" key={d.facility_id}>
                  <div className="ic ic-critical"><IconBuilding /></div>
                  <div className="bd">
                    <div className="t">{d.name} is down · <span className="amt">evac zone {d.evac_zone}</span></div>
                    <div className="d">Dialysis, infusion and OTP on site. Book alternate sites now; controlled-substance patients cannot use the retail refill route.</div>
                    <div className="act"><button className="sm" onClick={() => onOpen("careteam")}>Open the list <IconArrow /></button></div>
                  </div>
                </div>
              ))}
              {landfall && (
                <div className="insight flat">
                  <div className="ic ic-high"><IconDrop /></div>
                  <div className="bd">
                    <div className="t">Coastal flood warning {fmtDate(landfall.date)} · <span className="amt">{landfall.flood} ZIPs</span></div>
                    <div className="d">Evacuation zones 1–2 ordered. Basement and ground-floor veterans with mobility limits go first.</div>
                    <div className="act"><button className="sm" onClick={() => onOpen("map")}>Show on the map <IconArrow /></button></div>
                  </div>
                </div>
              )}
              {peakOutage > 0 && (
                <div className="insight flat">
                  <div className="ic ic-high"><IconBolt /></div>
                  <div className="bd"><div className="t">Outage in {peakOutage} ZIPs at peak</div><div className="d">Oxygen, ventilator and home-dialysis patients need a backup-power plan and a call within two hours of the outage.</div></div>
                </div>
              )}
              {firstHeat && (
                <div className="insight flat">
                  <div className="ic ic-medium"><IconSun /></div>
                  <div className="bd"><div className="t">Heat alert from {fmtDate(firstHeat.date)}</div><div className="d">Diuretic and anticholinergic loads, no AC, utility-shutoff risk: cooling-center rides are booked, not suggested.</div></div>
                </div>
              )}
              {peakMail > 0 && (
                <div className="insight flat">
                  <div className="ic ic-medium"><IconMail /></div>
                  <div className="bd"><div className="t">Mail delivery disrupted in {peakMail} ZIPs</div><div className="d">Four in five VA prescriptions arrive by mail. Early refills and local pickup for anyone under ten days of supply.</div></div>
                </div>
              )}
              {!down.length && !landfall && !firstHeat && (
                <div className="insight flat"><div className="ic ic-good"><IconAlert /></div><div className="bd"><div className="t">Quiet week</div><div className="d">No alerts in the window.</div></div></div>
              )}
            </div>
          </div>

          <div className="aiask">
            <div className="aiask-h"><span className="aidiamond">◆</span> Care-team morning brief <span className="aiask-tag">from the forecast</span></div>
            <div className="aiask-sub">Written from today's hazards, the site status table and the action list. No model call: every sentence is a number on this page.</div>
            <div className="brief"><p className="bt">{brief}</p></div>
            <div className="askchips">
              <span onClick={() => onOpen("command")}>Open the command center</span>
              <span onClick={() => onOpen("careteam")}>Today's action list</span>
              <span onClick={() => onOpen("report")}>How do we know it works?</span>
            </div>
          </div>
        </div>
      </div>

      <div>
        <div className="tbar"><h2>Top of today's list</h2><span className="ct">{actions.n_selected} actions at {callsCap} calls</span><span className="sp" /><button className="sm" onClick={() => onOpen("careteam")}>Full list <IconArrow /></button></div>
        <div className="tablewrap auto">
          <table>
            <thead><tr><th className="n">#</th><th>Veteran</th><th>Tier</th><th>Action</th><th>Owner</th><th className="n">Expected harm averted</th></tr></thead>
            <tbody>
              {actions.actions.slice(0, 8).map((a) => (
                <tr key={a.action_id} className="click" onClick={() => onOpen("careteam")}>
                  <td className="n">{a.rank}</td>
                  <td>{handleFor(a.name_display, a.veteran_id)}<span className="sub">{a.borough} · {a.modzcta}</span></td>
                  <td><span className={TIER_BADGE[a.tier]}>{TIER_LABEL[a.tier]}</span></td>
                  <td>{ACTION_LABEL[a.action] ?? a.action}</td>
                  <td className="muted">{a.owner.replace("_", " ")}</td>
                  <td className="n">{a.eha.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

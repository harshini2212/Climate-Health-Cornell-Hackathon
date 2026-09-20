/**
 * The at-risk patient list, in three tiers. Each row shows at a glance how many risk
 * factors flagged the patient and how many actions are suggested; hovering shows them;
 * clicking opens the chart. This is the clinician's workstation, so it shows names.
 *
 * Risk factors come from the veteran card (the posterior's own drivers per need), fetched
 * lazily for the rows on screen and cached, so the list itself renders at once.
 */

import { useEffect, useMemo, useState } from "react";
import { getVeteran } from "../lib/api";
import { ACTION_LABEL, NEED_SHORT, fmtInt, pct } from "../lib/labels";
import type { ActionRow, ActionsResponse, VeteranCard } from "../lib/types";
import type { VeteranFocus } from "../screens/Week";
import { IconArrow } from "./Icons";

const TIER3 = [
  { key: "act_now", label: "High risk", cls: "t-high", note: "act today", empty: "No veteran crosses the Act-now line on this day." },
  { key: "find_out", label: "Moderate", cls: "t-mod", note: "wide interval, find out",
    empty: "Find-out needs a wide interval. At rung 0 the prior-only model has almost no epistemic spread, so this tier fills once the fitted model lands." },
  { key: "self_serve", label: "Watch", cls: "t-watch", note: "self-serve, verified text", empty: "Nobody in this tier on this day." },
] as const;

interface Patient {
  veteranId: string;
  name: string;
  borough: string;
  modzcta: string;
  tier: string;
  actions: ActionRow[];
  eha: number;
}

function groupPatients(resp: ActionsResponse): Patient[] {
  const by = new Map<string, Patient>();
  for (const a of resp.actions) {
    const p = by.get(a.veteran_id) ?? { veteranId: a.veteran_id, name: a.name_display, borough: a.borough, modzcta: a.modzcta, tier: a.tier, actions: [], eha: 0 };
    p.actions.push(a);
    p.eha += a.eha;
    if (a.tier === "act_now") p.tier = "act_now";
    else if (a.tier === "find_out" && p.tier !== "act_now") p.tier = "find_out";
    by.set(a.veteran_id, p);
  }
  return [...by.values()].sort((a, b) => b.eha - a.eha);
}

/** Distinct driver phrases across the five needs, the card's "risk factors". */
function riskFactors(card: VeteranCard): string[] {
  const out: string[] = [];
  for (const n of card.needs) for (const d of n.drivers) if (!out.includes(d)) out.push(d);
  return out;
}

interface Props {
  resp: ActionsResponse | null;
  onOpen: (f: VeteranFocus) => void;
  /** Rows per tier before "show all". */
  limit?: number;
}

export function PatientList({ resp, onOpen, limit = 8 }: Props) {
  const [cards, setCards] = useState<Record<string, VeteranCard>>({});
  const [hover, setHover] = useState<string | null>(null);
  const [showAll, setShowAll] = useState<Record<string, boolean>>({});
  const patients = useMemo(() => (resp ? groupPatients(resp) : []), [resp]);
  const tiers = useMemo(() => TIER3.map((t) => ({ ...t, rows: patients.filter((p) => p.tier === t.key) })), [patients]);

  // Prefetch the cards for the rows on screen so the counts appear without a hover.
  useEffect(() => {
    if (!resp) return;
    const visible = tiers.flatMap((t) => t.rows.slice(0, showAll[t.key] ? 60 : limit));
    let live = true;
    (async () => {
      for (const p of visible) {
        if (cards[p.veteranId]) continue;
        try {
          const c = await getVeteran(p.veteranId, resp.date);
          if (!live) return;
          setCards((m) => (m[p.veteranId] ? m : { ...m, [p.veteranId]: c }));
        } catch {
          /* no card offline for this id; the row still renders */
        }
      }
    })();
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resp, tiers, showAll, limit]);

  if (!resp) return <div className="sk" style={{ height: 360 }} />;

  return (
    <div className="plist">
      {tiers.map((t) => {
        const rows = showAll[t.key] ? t.rows.slice(0, 60) : t.rows.slice(0, limit);
        return (
          <section key={t.key} className={`ptier ${t.cls}`}>
            <header>
              <span className="pt-dot" />
              <b>{t.label}</b>
              <span className="muted">{t.note}</span>
              <span className="sp" />
              <span className="pt-n">{fmtInt(t.rows.length)}</span>
            </header>
            {rows.length === 0 && <div className="muted small" style={{ padding: "9px 13px", lineHeight: 1.5 }}>{t.empty}</div>}
            {rows.map((p, i) => {
              const card = cards[p.veteranId];
              const rf = card ? riskFactors(card) : null;
              const top = card ? [...card.needs].sort((a, b) => b.p_mean - a.p_mean)[0] : null;
              return (
                <div key={p.veteranId} className={`prow${hover === p.veteranId ? " hov" : ""}`} onMouseEnter={() => setHover(p.veteranId)} onMouseLeave={() => setHover(null)} onClick={() => onOpen({ veteranId: p.veteranId, actionId: p.actions[0].action_id, date: resp.date })} role="button" tabIndex={0}>
                  <span className="pr-rank">{i + 1}</span>
                  <span className="pr-name"><b>{p.name}</b><span>{p.borough} · {p.modzcta}{top ? ` · ${NEED_SHORT[top.need]} ${pct(top.p_mean)}` : ""}</span></span>
                  <span className="pr-glance">
                    <span className={`gl${rf && rf.length >= 4 ? " hot" : ""}`} title="Risk factors flagged by the model"><b>{rf ? rf.length : "…"}</b> risk factor{rf && rf.length === 1 ? "" : "s"}</span>
                    <span className="gl" title="Suggested actions"><b>{p.actions.length}</b> action{p.actions.length === 1 ? "" : "s"}</span>
                  </span>
                  <span className="pr-action">{ACTION_LABEL[p.actions[0].action] ?? p.actions[0].action}</span>
                  <IconArrow />
                  {hover === p.veteranId && (
                    <div className="pop" onClick={(e) => e.stopPropagation()}>
                      <div className="pop-h"><b>{p.name}</b><span className="muted">{card ? `${card.age}, ${card.facility_name}` : "loading the chart…"}</span></div>
                      <div className="pop-sec">Risk factors</div>
                      {rf ? (rf.length ? <ul>{rf.slice(0, 6).map((d) => <li key={d}>{d}</li>)}</ul> : <div className="muted small">none flagged</div>) : <div className="sk" style={{ height: 40 }} />}
                      <div className="pop-sec">Suggested actions</div>
                      <ul>{p.actions.map((a) => <li key={a.action_id}><b>{ACTION_LABEL[a.action] ?? a.action}</b> <span className="muted">{a.headline}</span></li>)}</ul>
                      <div className="muted small" style={{ marginTop: 6 }}>Click to open the chart</div>
                    </div>
                  )}
                </div>
              );
            })}
            {t.rows.length > limit && (
              <button className="ghost sm" style={{ margin: "4px 8px 8px" }} onClick={() => setShowAll((s) => ({ ...s, [t.key]: !s[t.key] }))}>{showAll[t.key] ? "Show fewer" : `Show all ${fmtInt(Math.min(60, t.rows.length))}`}</button>
            )}
          </section>
        );
      })}
    </div>
  );
}

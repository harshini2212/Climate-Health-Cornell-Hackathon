/**
 * The reveal. The week board shows a handle; this is what opens when someone with a reason
 * to look clicks it, and it is the only place a name, a condition or a medicine appears.
 *
 * Five needs as 10-90% intervals, with the share of that spread which is model uncertainty
 * hatched inside the band: the wide, hatched ones are the "find out" cases, where a
 * three-minute call is worth more than a confident number. Drivers are the posterior's own
 * contributions in plain language; there is no SHAP here.
 *
 * Leeward never changes a medication. The medication panel exists so a VA clinical
 * pharmacist can decide, and `pharmacist_slot` is scarce because it is their afternoon.
 */

import { useEffect, useState } from "react";
import { IconBack, IconMessage } from "../components/Icons";
import { getVeteran } from "../lib/api";
import { ACTION_LABEL, NEED_LABEL, TIER_BADGE, TIER_LABEL, fmtDate, initialsOf, pct } from "../lib/labels";
import type { NeedScore, VeteranCard as Card } from "../lib/types";
import { MessagePanel } from "./Message";

const CAREGIVER_LABEL: Record<string, string> = {
  none: "No caregiver on record",
  informal_coresident: "Informal caregiver, same address",
  informal_remote: "Informal caregiver, lives elsewhere",
  va_pcafc: "VA PCAFC caregiver",
};

function NeedBar({ n, scale }: { n: NeedScore; scale: number }) {
  const left = (n.p_lo80 / scale) * 100;
  const width = Math.max(0.6, ((n.p_hi80 - n.p_lo80) / scale) * 100);
  const meanAt = (n.p_mean / scale) * 100;
  const epiWidth = width * Math.min(1, Math.max(0, n.p_epistemic_share));
  const epiLeft = Math.min(Math.max(meanAt - epiWidth / 2, left), left + width - epiWidth);
  return (
    <div className="needrow">
      <div className="pl">
        <span>{NEED_LABEL[n.need] ?? n.need}</span>
        <span><b>{pct(n.p_mean)}</b><small className="muted"> {pct(n.p_lo80)}–{pct(n.p_hi80)}</small></span>
      </div>
      <div className="needtrack">
        <div className="nb" style={{ left: `${left}%`, width: `${width}%` }} />
        <div className="ne" style={{ left: `${epiLeft}%`, width: `${epiWidth}%` }} />
        <div className="nm" style={{ left: `${meanAt}%` }} />
      </div>
      <div className="needfoot">{pct(n.p_epistemic_share)} of that spread is model uncertainty{n.drivers.length > 0 ? ` · ${n.drivers.join(" · ")}` : ""}</div>
    </div>
  );
}

function Meds({ m }: { m: Card["medications"] }) {
  const flags: { on: boolean; label: string; crit?: boolean }[] = [
    { on: m.combo_raas_diuretic, label: "ACE-i/ARB + diuretic (the pair CDC names)", crit: true },
    { on: m.renal_triple, label: "Renal triple whammy", crit: true },
    { on: m.cold_chain, label: "Cold chain" },
    { on: m.controlled, label: "Controlled substance" },
    { on: m.narrow_ti, label: "Narrow therapeutic index" },
    { on: m.mail_order_pharmacy, label: "Mail-order pharmacy" },
  ];
  return (
    <>
      <div className="proof">
        <div className="p"><div className="pl">Thermoregulatory</div><div className={`pv${m.thermoreg_score >= 2 ? " red" : ""}`}>{m.thermoreg_score.toFixed(1)}</div></div>
        <div className="p"><div className="pl">ACB burden</div><div className={`pv${m.acb_score >= 3 ? " red" : ""}`}>{m.acb_score}</div></div>
        <div className="p"><div className="pl">Active meds</div><div className="pv">{m.n_active_meds}</div></div>
        <div className="p"><div className="pl">Days of supply</div><div className={`pv${m.days_supply_remaining <= 14 ? " red" : ""}`}>{m.days_supply_remaining}</div></div>
      </div>
      <div className="chips" style={{ margin: "12px 0" }}>
        {flags.filter((f) => f.on).map((f) => <span key={f.label} className={`flagpill${f.crit ? " crit" : ""}`}>{f.label}</span>)}
        {flags.every((f) => !f.on) && <span className="tag">No medication flags</span>}
      </div>
      {m.notes.length > 0 && <ul className="mednotes">{m.notes.map((n) => <li key={n}>{n}</li>)}</ul>}
      <div className="callout" style={{ marginTop: 12 }}>Leeward never changes a medication. This flags the veteran for the VA clinical pharmacist, who decides.</div>
    </>
  );
}

interface Props {
  focus: { veteranId: string; actionId: string; date: string } | null;
  onOpenMessage: () => void;
  onBack: () => void;
}

export function VeteranScreen({ focus, onOpenMessage, onBack }: Props) {
  const [card, setCard] = useState<Card | null>(null);
  const [error, setError] = useState<string | null>(null);
  const veteranId = focus?.veteranId;
  const date = focus?.date;

  useEffect(() => {
    if (!veteranId || !date) return;
    let live = true;
    setCard(null);
    setError(null);
    getVeteran(veteranId, date)
      .then((c) => live && setCard(c))
      .catch((e) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [veteranId, date]);

  if (!focus) {
    return (
      <div className="page">
        <div className="empty">
          <div className="eh">Pick someone first</div>
          Open a row on the week board or the action list. Names and conditions are deliberately not browsable from the nav; this screen needs a reason to be here.
        </div>
      </div>
    );
  }
  if (error) return <div className="page"><div className="empty"><div className="eh">Card unavailable</div>{error}</div></div>;
  if (!card) return <div className="page dash"><div className="sk" style={{ height: 96 }} /><div className="row g2"><div className="sk" style={{ height: 360 }} /><div className="sk" style={{ height: 360 }} /></div></div>;

  const scale = Math.max(0.25, ...card.needs.map((n) => n.p_hi80)) * 1.05;

  return (
    <div className="page dash">
      <button className="backbtn" onClick={onBack}><IconBack /> Back to the week board</button>

      <div className="card">
        <div className="vhead">
          <div className="av">{initialsOf(card.name_display)}</div>
          <div>
            <div className="nm">{card.name_display}</div>
            <div className="meta">
              <span>{card.age} years</span><span>·</span><span>{card.borough} {card.modzcta}</span><span>·</span><span>{card.facility_name} (station {card.facility_id})</span><span>·</span><span>{fmtDate(card.date)}</span>
            </div>
          </div>
          <div className="right">
            <span className={TIER_BADGE[card.tier]}>{TIER_LABEL[card.tier] ?? card.tier}</span>
            <button className="sm" onClick={onOpenMessage}><IconMessage /> Message</button>
          </div>
        </div>
        <div className="callout" style={{ marginTop: 14 }}>{card.why_this_tier}</div>
      </div>

      <div className="row g2">
        <div className="card">
          <div className="ch"><h3>Need this week</h3><span className="sub">chance the need arises, with its 10–90% interval</span></div>
          <div className="prows">{card.needs.map((n) => <NeedBar key={n.need} n={n} scale={scale} />)}</div>
          <div className="legend">
            <span><i style={{ background: "var(--accentbd)", height: 8 }} /> 10–90% interval</span>
            <span><i className="hatch" /> model uncertainty</span>
            <span><i style={{ background: "var(--accent2)", width: 3, height: 12 }} /> mean</span>
          </div>
        </div>

        <div className="stack">
          <div className="card">
            <div className="ch"><h3>Context</h3></div>
            <div className="facts">
              <div><span className="k">Conditions</span><span className="v">{card.conditions.join(", ") || "none recorded"}</span></div>
              <div><span className="k">Powered equipment</span><span className="v">{card.powered_equipment.replace(/_/g, " ")}</span></div>
              <div><span className="k">Caregiver</span><span className="v">{CAREGIVER_LABEL[card.caregiver] ?? card.caregiver}</span></div>
              <div><span className="k">Housing</span><span className="v">{card.floor} floor · evacuation zone {card.evac_zone || "none"}</span></div>
            </div>
          </div>
          <div className="card">
            <div className="ch"><h3>Plan</h3><span className="sub">what is queued for this veteran</span></div>
            <ol className="plan">
              {card.planned_actions.map((a) => <li key={a}>{ACTION_LABEL[a] ?? a}</li>)}
              {card.planned_actions.length === 0 && <li className="muted">Nothing queued</li>}
            </ol>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="ch"><h3>Medication</h3><span className="sub">from the RxNorm codes on the record, mapped to the VA's own drug classes</span></div>
        <Meds m={card.medications} />
      </div>

      <div className="sec"><h2>Outreach</h2></div>
      <MessagePanel actionId={focus.actionId} />

      {card.is_synthetic && <p className="muted small" style={{ margin: 0 }}>Synthetic person. Neighbourhood rates, medication classes and facility status are real; this veteran is not.</p>}
    </div>
  );
}

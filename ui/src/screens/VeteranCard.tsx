/**
 * The reveal. The week board shows a handle; this is what opens when someone with a reason
 * to look clicks it, and it is the only place a name, a condition or a medicine appears.
 *
 * Five needs as 10-90% intervals, with the share of that spread which is model uncertainty
 * shaded inside the band -- the wide, hatched ones are the "find out" cases, where a
 * three-minute call is worth more than a confident number. Drivers are the posterior's own
 * contributions in plain language; there is no SHAP here.
 *
 * Leeward never changes a medication. The medication panel exists so a VA clinical
 * pharmacist can decide, and `pharmacist_slot` is scarce because it is their afternoon.
 */

import { useEffect, useState } from "react";
import { getVeteran } from "../lib/api";
import { ACTION_LABEL, NEED_LABEL, TIER_BADGE, TIER_LABEL } from "../lib/labels";
import type { NeedScore, VeteranCard as Card } from "../lib/types";
import { MessagePanel } from "./Message";

const CAREGIVER_LABEL: Record<string, string> = {
  none: "No caregiver on record",
  informal_coresident: "Informal caregiver, same address",
  informal_remote: "Informal caregiver, lives elsewhere",
  va_pcafc: "VA PCAFC caregiver",
};

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

/**
 * One need. The band runs lo80 -> hi80 on a shared scale; the hatched inner band is the
 * epistemic share of that spread, so "we do not know" is visible rather than averaged away.
 */
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
        <span>
          <b>{pct(n.p_mean)}</b>
          <small className="muted"> {pct(n.p_lo80)}–{pct(n.p_hi80)}</small>
        </span>
      </div>
      <div className="needtrack">
        <div className="nb" style={{ left: `${left}%`, width: `${width}%` }} />
        <div className="ne" style={{ left: `${epiLeft}%`, width: `${epiWidth}%` }} />
        <div className="nm" style={{ left: `${meanAt}%` }} />
      </div>
      <div className="muted needfoot">
        {pct(n.p_epistemic_share)} of that spread is model uncertainty
      </div>
      {n.drivers.length > 0 && (
        <div className="chips">
          {n.drivers.map((d) => <span key={d} className="chip">{d}</span>)}
        </div>
      )}
    </div>
  );
}

function Meds({ m }: { m: Card["medications"] }) {
  const flags: { on: boolean; label: string }[] = [
    { on: m.combo_raas_diuretic, label: "RAAS + diuretic (the combination CDC names)" },
    { on: m.renal_triple, label: "Renal triple whammy" },
    { on: m.cold_chain, label: "Cold chain" },
    { on: m.controlled, label: "Controlled" },
    { on: m.narrow_ti, label: "Narrow therapeutic index" },
    { on: m.mail_order_pharmacy, label: "Mail-order pharmacy" },
  ];
  return (
    <>
      <div className="proof">
        <div className="p"><div className="pl">Thermoregulatory</div><div className="pv">{m.thermoreg_score.toFixed(1)}</div></div>
        <div className="p"><div className="pl">ACB</div><div className={`pv${m.acb_score >= 3 ? " red" : ""}`}>{m.acb_score}</div></div>
        <div className="p"><div className="pl">Active meds</div><div className="pv">{m.n_active_meds}</div></div>
        <div className="p"><div className="pl">Days of supply</div><div className={`pv${m.days_supply_remaining <= 14 ? " red" : ""}`}>{m.days_supply_remaining}</div></div>
      </div>
      <div className="chips" style={{ margin: "10px 0" }}>
        {flags.filter((f) => f.on).map((f) => <span key={f.label} className="flagpill">{f.label}</span>)}
        {flags.every((f) => !f.on) && <span className="tag">No medication flags</span>}
      </div>
      {m.notes.length > 0 && (
        <ul className="mednotes">{m.notes.map((n) => <li key={n}>{n}</li>)}</ul>
      )}
      <div className="guard">
        Leeward never changes a medication. This flags the veteran for the VA clinical
        pharmacist, who decides.
      </div>
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
          Open a card on the week board. Names and conditions are deliberately not
          browsable from the nav — this screen needs a reason to be here.
        </div>
      </div>
    );
  }

  if (error) {
    return <div className="page"><div className="empty"><div className="eh">Card unavailable</div>{error}</div></div>;
  }
  if (!card) return <div className="page"><div className="spin">Loading the card…</div></div>;

  const scale = Math.max(0.25, ...card.needs.map((n) => n.p_hi80)) * 1.05;

  return (
    <div className="page dash">
      <span className="backbtn" onClick={onBack}>← Back to the week board</span>

      <div className="card">
        <div className="ch">
          <div>
            <h3 style={{ fontSize: 20 }}>{card.name_display}</h3>
            <div className="muted" style={{ fontSize: 13 }}>
              {card.age} · {card.borough} {card.modzcta} · {card.facility_name} ({card.facility_id})
            </div>
          </div>
          <div className="sp" />
          <span className={TIER_BADGE[card.tier]}>{TIER_LABEL[card.tier] ?? card.tier}</span>
        </div>
        <div className="guard" style={{ marginTop: 0 }}>{card.why_this_tier}</div>
      </div>

      <div className="row g2">
        <div className="card">
          <div className="ch"><h3>Need this week</h3></div>
          <div className="prows">
            {card.needs.map((n) => <NeedBar key={n.need} n={n} scale={scale} />)}
          </div>
          <div className="legend">
            <span><i style={{ background: "var(--accentbd)", height: 8 }} /> 10–90% interval</span>
            <span><i className="hatch" /> model uncertainty</span>
            <span><i style={{ background: "var(--accent)", width: 3, height: 12 }} /> mean</span>
          </div>
        </div>

        <div className="card">
          <div className="ch"><h3>Context</h3></div>
          <div className="facts">
            <div><span className="k">Conditions</span><span className="v">{card.conditions.join(", ") || "none recorded"}</span></div>
            <div><span className="k">Powered equipment</span><span className="v">{card.powered_equipment.replace(/_/g, " ")}</span></div>
            <div><span className="k">Caregiver</span><span className="v">{CAREGIVER_LABEL[card.caregiver] ?? card.caregiver}</span></div>
            <div><span className="k">Housing</span><span className="v">{card.floor} floor, evacuation zone {card.evac_zone || "—"}</span></div>
          </div>
          <div className="ch" style={{ marginTop: 16 }}><h3>Planned actions</h3></div>
          <div className="chips">
            {card.planned_actions.map((a) => <span key={a} className="chip">{ACTION_LABEL[a] ?? a}</span>)}
            {card.planned_actions.length === 0 && <span className="tag">None queued</span>}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="ch">
          <h3>Medication</h3>
          <div className="sp" />
          <button className="sm" onClick={onOpenMessage}>Open the message</button>
        </div>
        <Meds m={card.medications} />
      </div>

      <h2>Outreach</h2>
      <MessagePanel actionId={focus.actionId} />

      {card.is_synthetic && (
        <p className="muted" style={{ fontSize: 12 }}>
          Synthetic person. Neighbourhood rates, medication classes and facility status are
          real; this veteran is not.
        </p>
      )}
    </div>
  );
}

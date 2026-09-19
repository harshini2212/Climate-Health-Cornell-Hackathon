/**
 * The reveal. The week board shows a handle; this is what opens when someone with a reason
 * to look clicks it, and it is the only place a name, a condition or a medicine appears.
 *
 * Five needs as 10-90% intervals, with the share of that spread which is model uncertainty
 * shaded inside the band — the wide-and-pale ones are the "find out" cases, where a
 * three-minute call is worth more than a confident number. Drivers are the posterior's own
 * contributions in plain language; there is no SHAP here.
 *
 * Leeward never changes a medication. The medication panel exists so a VA clinical
 * pharmacist can decide, and `pharmacist_slot` is scarce because it is their afternoon.
 */

import { useEffect, useState } from "react";
import { getVeteran } from "../lib/api";
import { TIER_LABEL, TIER_STEP } from "../lib/colors";
import type { ActionRow, NeedScore, VeteranCard as Card } from "../lib/types";
import { MessagePanel } from "./Message";

const NEED_LABEL: Record<string, string> = {
  breathing: "Breathing crisis",
  heat: "Heat illness",
  mental: "Mental-health crisis",
  treatment_gap: "Treatment gap",
  access_loss: "Loss of access to care",
};

const ACTION_LABEL: Record<string, string> = {
  care_team_call: "Care-team call",
  check_in_call: "3-minute check-in",
  backup_power_plan: "Backup-power plan",
  cold_chain_plan: "Cold-chain plan",
  early_refill: "Early refill",
  switch_to_local_pickup: "Switch to local pickup",
  cooling_center_ride: "Cooling-centre ride",
  clean_air_room: "Clean-air room",
  alt_site_booking: "Alternate-site booking",
  evacuation_assist: "Evacuation assist",
  assign_buddy: "Assign buddy",
  pharmacist_med_review: "Pharmacist review",
  controlled_substance_bridge: "Controlled-substance bridge",
  heap_application: "HEAP application",
  verified_text: "Verified text",
};

const CAREGIVER_LABEL: Record<string, string> = {
  none: "No caregiver on record",
  informal_coresident: "Informal caregiver, same address",
  informal_remote: "Informal caregiver, lives elsewhere",
  va_pcafc: "VA PCAFC caregiver",
};

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

/**
 * One need. The band runs lo80 -> hi80 on a shared scale; the shaded inner band is the
 * epistemic share of that spread, so "we do not know" is visible rather than averaged away.
 */
function NeedBar({ n, scale }: { n: NeedScore; scale: number }) {
  const left = (n.p_lo80 / scale) * 100;
  const width = Math.max(0.6, ((n.p_hi80 - n.p_lo80) / scale) * 100);
  const meanAt = (n.p_mean / scale) * 100;
  const epiWidth = width * Math.min(1, Math.max(0, n.p_epistemic_share));
  const epiLeft = Math.min(Math.max(meanAt - epiWidth / 2, left), left + width - epiWidth);
  return (
    <div className="need">
      <div className="need-head">
        <span className="need-name">{NEED_LABEL[n.need] ?? n.need}</span>
        <span className="need-num">
          {pct(n.p_mean)}
          <small>
            {pct(n.p_lo80)}–{pct(n.p_hi80)}
          </small>
        </span>
      </div>
      <div className="need-track">
        <div className="need-band" style={{ left: `${left}%`, width: `${width}%` }} />
        <div className="need-epi" style={{ left: `${epiLeft}%`, width: `${epiWidth}%` }} />
        <div className="need-mean" style={{ left: `${meanAt}%` }} />
      </div>
      <div className="need-foot">
        {pct(n.p_epistemic_share)} of that spread is model uncertainty
      </div>
      {n.drivers.length > 0 && (
        <div className="chips">
          {n.drivers.map((d) => (
            <span key={d} className="chip">{d}</span>
          ))}
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
    <div className="meds">
      <div className="med-nums">
        <div>
          <span className="k">Thermoregulatory</span>
          <span className="v">{m.thermoreg_score.toFixed(1)}</span>
        </div>
        <div>
          <span className="k">ACB</span>
          <span className={`v${m.acb_score >= 3 ? " warn" : ""}`}>{m.acb_score}</span>
        </div>
        <div>
          <span className="k">Active meds</span>
          <span className="v">{m.n_active_meds}</span>
        </div>
        <div>
          <span className="k">Days of supply</span>
          <span className={`v${m.days_supply_remaining <= 14 ? " warn" : ""}`}>
            {m.days_supply_remaining}
          </span>
        </div>
      </div>
      <div className="chips">
        {flags.filter((f) => f.on).map((f) => (
          <span key={f.label} className="chip warn">{f.label}</span>
        ))}
        {flags.every((f) => !f.on) && <span className="chip">No medication flags</span>}
      </div>
      {m.notes.length > 0 && (
        <ul className="med-notes">
          {m.notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
      )}
      <p className="med-guard">
        Leeward never changes a medication. This flags the veteran for the VA clinical
        pharmacist, who decides.
      </p>
    </div>
  );
}

interface Props {
  veteranId: string;
  date: string;
  /** The queue card that was clicked, so the panel can open its message. */
  action?: ActionRow | null;
  onClose: () => void;
}

export function VeteranPanel({ veteranId, date, action, onClose }: Props) {
  const [card, setCard] = useState<Card | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
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

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const scale = card
    ? Math.max(0.25, ...card.needs.map((n) => n.p_hi80)) * 1.05
    : 1;

  return (
    <div className="drawer-scrim" onClick={onClose}>
      <aside className="drawer" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Veteran card">
        <div className="drawer-head">
          <div>
            {card ? (
              <>
                <h2>{card.name_display}</h2>
                <div className="sub">
                  {card.age} · {card.borough} {card.modzcta} · {card.facility_name} ({card.facility_id})
                </div>
              </>
            ) : (
              <h2>{error ? "Card unavailable" : "Loading…"}</h2>
            )}
          </div>
          <button className="close" onClick={onClose} aria-label="Close">✕</button>
        </div>

        {error && <div className="msg-empty">{error}</div>}

        {card && (
          <div className="drawer-body">
            <div className="tier-line" style={{ borderLeftColor: TIER_STEP[card.tier] }}>
              <strong>{TIER_LABEL[card.tier] ?? card.tier}</strong>
              {card.why_this_tier}
            </div>

            <section>
              <h3>Need this week</h3>
              {card.needs.map((n) => <NeedBar key={n.need} n={n} scale={scale} />)}
            </section>

            <section>
              <h3>Context</h3>
              <div className="facts">
                <div><span className="k">Conditions</span><span className="v">{card.conditions.join(", ") || "none recorded"}</span></div>
                <div><span className="k">Powered equipment</span><span className="v">{card.powered_equipment.replace(/_/g, " ")}</span></div>
                <div><span className="k">Caregiver</span><span className="v">{CAREGIVER_LABEL[card.caregiver] ?? card.caregiver}</span></div>
                <div><span className="k">Housing</span><span className="v">{card.floor} floor, evacuation zone {card.evac_zone || "—"}</span></div>
              </div>
            </section>

            <section>
              <h3>Medication</h3>
              <Meds m={card.medications} />
            </section>

            <section>
              <h3>Planned actions</h3>
              <div className="chips">
                {card.planned_actions.map((a) => (
                  <span key={a} className="chip">{ACTION_LABEL[a] ?? a}</span>
                ))}
                {card.planned_actions.length === 0 && <span className="chip">None queued</span>}
              </div>
            </section>

            {action?.action_id && (
              <section>
                <h3>Outreach</h3>
                <MessagePanel actionId={action.action_id} />
              </section>
            )}

            {card.is_synthetic && (
              <p className="synthetic-note">
                Synthetic person. Neighbourhood rates, medication classes and facility
                status are real; this veteran is not.
              </p>
            )}
          </div>
        )}
      </aside>
    </div>
  );
}

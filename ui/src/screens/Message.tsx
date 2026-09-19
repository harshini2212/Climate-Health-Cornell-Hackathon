/**
 * The outreach text, with the five things that make it checkable rendered as a visible
 * checklist beside it. The checklist is not decoration: a veteran is being asked to trust
 * a message about their own care during a storm, and a scammer sending a lookalike cannot
 * produce the VA channel tag or the phrase the caller reads back.
 *
 * Each row is verified against the body we are actually showing, not against the flag the
 * API set, so the checklist cannot go green on a message that lost a line in transit.
 */

import { useEffect, useState } from "react";
import { IconShield } from "../components/Icons";
import { getMessage } from "../lib/api";
import type { Message } from "../lib/types";

const CHANNEL_LABEL: Record<string, string> = {
  VEText: "VEText — the VA's own SMS channel",
  MHV: "My HealtheVet secure message",
  care_team_phone: "Care-team phone, from a VA number",
};

interface Check {
  label: string;
  ok: boolean;
  found: string;
}

function checks(m: Message): Check[] {
  const body = m.body;
  const phrase = m.verification_phrase ?? "";
  return [
    {
      label: "VA channel tag",
      ok: body.includes(`[VA ${m.channel}]`),
      found: CHANNEL_LABEL[m.channel] ?? m.channel,
    },
    {
      label: "Four-word verification phrase",
      ok: phrase.trim().split(/\s+/).length === 4 && body.includes(phrase),
      found: `“${phrase}” — ask the caller to say it`,
    },
    {
      label: "Never-pay line",
      ok: body.includes("The VA will never ask you to pay, wire money, or share bank details"),
      found: "The VA will never ask you to pay, wire money, or share bank details",
    },
    {
      label: "VSAFE, to report a scam",
      ok: body.includes("833-388-7233"),
      found: "VSAFE 833-388-7233",
    },
    {
      label: "Veterans Crisis Line",
      ok: body.includes("988"),
      found: "Dial 988, press 1",
    },
  ];
}

/** The panel, reusable inside the veteran card as well as on its own screen. */
export function MessagePanel({ actionId }: { actionId: string }) {
  const [msg, setMsg] = useState<Message | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setMsg(null);
    setError(null);
    getMessage(actionId)
      .then((m) => live && setMsg(m))
      .catch((e) => live && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      live = false;
    };
  }, [actionId]);

  if (error) return <div className="spin">No message for this action ({error}).</div>;
  if (!msg) return <div className="spin">Loading the message…</div>;

  const rows = checks(msg);
  const allOk = rows.every((r) => r.ok);

  return (
    <div className="row g2 msgwrap">
      <div className="card">
        <div className="ch">
          <h3>What the veteran receives</h3>
          <div className="sp" />
          <span className="tag">{msg.channel}</span>
          <span className="tag">to the {msg.addressed_to}</span>
        </div>
        <pre className="msgbody">{msg.body}</pre>
      </div>
      <div className="card">
        <div className="ch">
          <IconShield />
          <h3>{allOk ? "All five elements present" : "This message is incomplete"}</h3>
        </div>
        <ul className="checks">
          {rows.map((r) => (
            <li key={r.label} className={r.ok ? "ok" : "bad"}>
              <span className="mk" aria-hidden="true">{r.ok ? "✓" : "✕"}</span>
              <span>
                <b>{r.label}</b>
                <small>{r.found}</small>
              </span>
            </li>
          ))}
        </ul>
        <p className="muted" style={{ fontSize: 12.5, marginBottom: 0 }}>
          Checked against the text on the left, not against the flags the API set.
        </p>
        {msg.scam_card_url && (
          <a href={msg.scam_card_url} target="_blank" rel="noreferrer" style={{ fontSize: 12.5 }}>
            VA scam-prevention card
          </a>
        )}
      </div>
    </div>
  );
}

interface ScreenProps {
  focus: { veteranId: string; actionId: string; date: string } | null;
  onBack: () => void;
}

/** The nav entry. Without a subject there is nothing honest to show, so it says so. */
export function MessageScreen({ focus, onBack }: ScreenProps) {
  if (!focus) {
    return (
      <div className="page">
        <div className="empty">
          <div className="eh">Pick someone first</div>
          Open a card on the week board; this screen shows the message that would go to
          that veteran, with its verification checklist.
        </div>
      </div>
    );
  }
  return (
    <div className="page dash">
      <span className="backbtn" onClick={onBack}>← Back to the card</span>
      <MessagePanel actionId={focus.actionId} />
    </div>
  );
}

/**
 * Ask Leeward: a question box that answers from the data on screen and the resource
 * library. Deterministic and offline by design; the footer says so.
 */

import { useEffect, useRef, useState } from "react";
import { IconArrow } from "./Icons";
import { SUGGESTED, answer, type Answer, type AskContext } from "../lib/ask";

interface Turn {
  role: "user" | "leeward";
  text: string;
  answer?: Answer;
}

interface Props {
  ctx: AskContext;
  onGoto?: (screen: string, date?: string) => void;
  /** Compact: fewer suggestions, shorter transcript, for the dashboard tile. */
  compact?: boolean;
}

export function AskPanel({ ctx, onGoto, compact = false }: Props) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [thinking, setThinking] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, thinking]);

  const ask = (text: string) => {
    const t = text.trim();
    if (!t) return;
    setTurns((ts) => [...ts, { role: "user", text: t }]);
    setQ("");
    setThinking(true);
    // A short beat so the answer reads as an answer, not a search result.
    setTimeout(() => {
      const a = answer(t, ctx);
      setTurns((ts) => [...ts, { role: "leeward", text: a.text, answer: a }]);
      setThinking(false);
    }, 450);
  };

  const suggestions = compact ? SUGGESTED.slice(0, 3) : SUGGESTED;

  return (
    <div className={`ask${compact ? " compact" : ""}`}>
      <div className="ask-h"><span className="aidiamond">◆</span> Ask Leeward <span className="aiask-tag">offline · from the data on screen</span></div>
      <div className="ask-log">
        {turns.length === 0 && (
          <div className="ask-empty">Ask about today's list, a site closure, capacity, an outage, or anything in the resource library.</div>
        )}
        {turns.map((t, i) => (
          <div key={i} className={`msg ${t.role}`}>
            {t.role === "leeward" && <span className="msg-dia">◆</span>}
            <div className="msg-body">
              <div>{t.text}</div>
              {t.answer?.bullets && <ul>{t.answer.bullets.map((b, k) => <li key={k}>{b}</li>)}</ul>}
              {t.answer?.resources && t.answer.resources.length > 0 && (
                <div className="msg-res">
                  {t.answer.resources.map((r) => <span key={r.id} className="chip" onClick={() => onGoto?.("library")}>{r.title}</span>)}
                </div>
              )}
              {t.answer?.goto && onGoto && (
                <button className="sm" style={{ marginTop: 8 }} onClick={() => onGoto(t.answer!.goto!.screen, t.answer!.goto!.date)}>{t.answer.goto.label} <IconArrow /></button>
              )}
            </div>
          </div>
        ))}
        {thinking && <div className="msg leeward"><span className="msg-dia">◆</span><div className="msg-body typing"><i /><i /><i /></div></div>}
        <div ref={endRef} />
      </div>
      <div className="askchips">
        {suggestions.map((s) => <span key={s} onClick={() => ask(s)}>{s}</span>)}
      </div>
      <form className="ask-in" onSubmit={(e) => { e.preventDefault(); ask(q); }}>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about the week, a patient group, a medication, a program…" aria-label="Ask Leeward" />
        <button type="submit" className="ai sm">Ask</button>
      </form>
    </div>
  );
}

/**
 * The resource library: climate-and-health guidance by event and topic, every card with
 * its source. Searchable, filterable, and the same content Ask Leeward answers from.
 */

import { useMemo, useState } from "react";
import { IconSearch } from "../components/Icons";
import { EVENT_LABEL, TOPIC_LABEL, searchLibrary, type EventKind, type Resource, type Topic } from "../lib/library";

const EVENTS: EventKind[] = ["all", "heat", "smoke", "flood", "outage"];
const TOPICS: Topic[] = ["medications", "conditions", "programs", "outreach", "housing"];

export function ResourceCard({ r, open, onToggle }: { r: Resource; open: boolean; onToggle: () => void }) {
  return (
    <div className={`res ev-${r.event}${open ? " open" : ""}`} onClick={onToggle} role="button" tabIndex={0}>
      <div className="res-top"><span className={`evtag ev-${r.event}`}>{EVENT_LABEL[r.event]}</span><span className="tag">{TOPIC_LABEL[r.topic]}</span></div>
      <div className="res-title">{r.title}</div>
      <div className="res-sum">{r.summary}</div>
      {open && <ul className="res-points">{r.points.map((p, i) => <li key={i}>{p}</li>)}</ul>}
      <div className="res-foot"><span>{r.points.length} points</span><span className="res-src">{r.source}</span></div>
    </div>
  );
}

export function Library({ initialEvent = "all" }: { initialEvent?: EventKind }) {
  const [q, setQ] = useState("");
  const [event, setEvent] = useState<EventKind>(initialEvent);
  const [topic, setTopic] = useState<Topic | undefined>(undefined);
  const [open, setOpen] = useState<string | null>(null);
  const rows = useMemo(() => searchLibrary(q, event, topic), [q, event, topic]);

  return (
    <div className="page dash">
      <div className="lib-bar">
        <div className="search lib-search"><IconSearch /><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search: diuretics, dialysis, oxygen, scams…" aria-label="Search the library" /></div>
        <div className="ctrls">
          {EVENTS.map((e) => <span key={e} className={`iv${event === e ? " on" : ""}`} onClick={() => setEvent(e)}>{EVENT_LABEL[e]}</span>)}
        </div>
        <div className="ctrls">
          <span className={`iv${!topic ? " on" : ""}`} onClick={() => setTopic(undefined)}>All topics</span>
          {TOPICS.map((t) => <span key={t} className={`iv${topic === t ? " on" : ""}`} onClick={() => setTopic(t)}>{TOPIC_LABEL[t]}</span>)}
        </div>
      </div>
      {rows.length === 0 ? (
        <div className="empty"><div className="eh">Nothing matches</div>Try a medication, a condition, or an event.</div>
      ) : (
        <div className="lib-grid">
          {rows.map((r) => <ResourceCard key={r.id} r={r} open={open === r.id} onToggle={() => setOpen(open === r.id ? null : r.id)} />)}
        </div>
      )}
      <p className="muted small" style={{ margin: 0 }}>Clinician-facing guidance for the care team, each card with its public source. Not advice to a patient; Leeward never changes a medication.</p>
    </div>
  );
}

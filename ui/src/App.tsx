import { useCallback, useEffect, useState, type ReactNode } from "react";
import { IconBook, IconCalendar, IconChart, IconHome, IconList, IconMap, IconMessage, IconSearch, IconSparkle, IconUser } from "./components/Icons";
import { getForecast, postActions, type Source } from "./lib/api";
import { DEFAULT_SCENARIO } from "./lib/config";
import { CHOICES, INITIAL_CHOICE } from "./lib/demo";
import { RUNG_LABEL, fmtDate } from "./lib/labels";
import { DEFAULT_CAPACITY } from "./lib/types";
import { Ask } from "./screens/Ask";
import { CareTeam } from "./screens/CareTeam";
import { CommandCenter, type VeteranFocus } from "./screens/CommandCenter";
import { Forecast } from "./screens/Forecast";
import { Library } from "./screens/Library";
import { Map } from "./screens/Map";
import { MessageScreen } from "./screens/Message";
import { Report } from "./screens/Report";
import { VeteranScreen } from "./screens/VeteranCard";

/**
 * The shell: a 230px sidebar with grouped nav, a sticky topbar, a page header, then the
 * page. Screens sit in the demo's order: Command center -> Forecast -> Map -> Action list ->
 * Veteran card -> Message -> Model report. Every screen opens on the same morning: the
 * window the API picks (two days before the first alert) or the calm week, from the
 * picker in the topbar.
 */

export type ScreenKey = "command" | "forecast" | "map" | "careteam" | "veteran" | "message" | "ask" | "library" | "report";

interface Screen {
  key: ScreenKey;
  group: string;
  label: string;
  title: string;
  sub: string;
  icon: ReactNode;
}

const SCREENS: Screen[] = [
  { key: "command", group: "Overview", label: "Command center", icon: <IconCalendar />, title: "Command center", sub: "The week ahead, who to reach, and what to know before you call" },
  { key: "forecast", group: "Overview", label: "Forecast", icon: <IconHome />, title: "Forecast", sub: "Hazards, sites and the panel over the next seven days" },
  { key: "map", group: "Overview", label: "Map", icon: <IconMap />, title: "Map", sub: "Expected need by ZIP, VA sites, and where the hazard lands" },
  { key: "careteam", group: "Care team", label: "Action list", icon: <IconList />, title: "Today's action list", sub: "Ranked by expected harm averted, cut at the team's real capacity" },
  { key: "veteran", group: "Care team", label: "Veteran card", icon: <IconUser />, title: "Veteran card", sub: "Five needs with their uncertainty, the drivers, the plan" },
  { key: "message", group: "Care team", label: "Message", icon: <IconMessage />, title: "Verified message", sub: "VA channel, four-word phrase, and the never-pay line" },
  { key: "ask", group: "Knowledge", label: "Ask Leeward", icon: <IconSparkle />, title: "Ask Leeward", sub: "Questions about the week, a patient group, a medication or a program" },
  { key: "library", group: "Knowledge", label: "Resource library", icon: <IconBook />, title: "Resource library", sub: "Climate and health guidance by event and topic, every card with its source" },
  { key: "report", group: "Proof", label: "Model report", icon: <IconChart />, title: "Model report", sub: "How we know it works: harm averted, recovery, calibration, fairness" },
];

export default function App() {
  const [screen, setScreen] = useState<ScreenKey>("command");
  const scenario = DEFAULT_SCENARIO.key;
  const [source, setSource] = useState<Source>("fixture");
  const [actNow, setActNow] = useState<number | null>(null);
  const [rung, setRung] = useState<number | null>(null);
  /** Which beat the demo is on: the opening window the API picks, or the calm week. */
  const [choice, setChoice] = useState(INITIAL_CHOICE);
  const startDay = CHOICES.find((c) => c.key === choice)?.day;
  /** The first day of that window, once the forecast has said what it is. */
  const [today, setToday] = useState<string | null>(null);
  /** Set when a day on the week ribbon is opened, so the action list starts on that date. */
  const [day, setDay] = useState<string | null>(null);
  /** Set when a queue row is clicked, so the Veteran card and Message screens have a subject. */
  const [focus, setFocus] = useState<VeteranFocus | null>(null);
  const onSource = useCallback((s: Source) => setSource(s), []);
  /** Present mode: no sidebar, no topbar, the page scaled for the big screen. P toggles, Esc exits. */
  const [present, setPresent] = useState(false);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.key === "p" || e.key === "P") setPresent((v) => !v);
      if (e.key === "Escape") setPresent(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    let live = true;
    setDay(null);
    getForecast(scenario, startDay)
      .then((f) => {
        if (!live) return null;
        setToday(f.dates[0] ?? null);
        return postActions({ date: f.dates[0], capacity: { ...DEFAULT_CAPACITY }, scenario });
      })
      .then((r) => {
        if (!r || !live) return;
        setActNow(r.counts_by_tier.act_now ?? 0);
        setRung(r.model_rung);
      })
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [scenario, startDay]);

  /** Ask and the dashboard both jump to a screen, sometimes on a particular day. */
  const goto = useCallback((s: ScreenKey, d?: string) => {
    if (d) setDay(d);
    setScreen(s);
  }, []);
  const openVeteran = useCallback((f: VeteranFocus) => {
    setFocus(f);
    setScreen("veteran");
  }, []);

  const current = SCREENS.find((s) => s.key === screen)!;
  const groups = [...new Set(SCREENS.map((s) => s.group))];

  return (
    <div className={`app${present ? " present" : ""}`}>
      <button className="present-btn primary sm" onClick={() => setPresent((v) => !v)} title="Toggle present mode (P)">{present ? "Exit present (Esc)" : "Present"}</button>
      <aside className="sidebar">
        <div className="brand">
          <div className="logo">L</div>
          <div className="nm">Leeward</div>
        </div>
        <nav className="nav">
          {groups.map((g) => (
            <div key={g} style={{ display: "contents" }}>
              <span className="grp">{g}</span>
              {SCREENS.filter((s) => s.group === g).map((s) => (
                <a key={s.key} className={s.key === screen ? "on" : ""} onClick={() => setScreen(s.key)} role="tab" aria-selected={s.key === screen} tabIndex={0}>
                  {s.icon}
                  <span>{s.label}</span>
                  {s.key === "careteam" && actNow ? <span className="cnt">{actNow}</span> : null}
                </a>
              ))}
            </div>
          ))}
        </nav>
        <div className="sidefoot">
          <div className="sideuser">
            <div className="avatar" title="Signed in">CT</div>
            <select title="Viewing as" defaultValue="care_team">
              <option value="care_team">VA care team</option>
              <option value="pharmacist">Clinical pharmacist</option>
              <option value="partner">Partner org</option>
            </select>
          </div>
        </div>
      </aside>

      <div className="mainwrap">
        <div className="topbar">
          <div className="search">
            <IconSearch />
            <span>Search veterans, ZIPs, sites…</span>
            <span className="kbd">⌘K</span>
          </div>
          <div className="sp" />
          <div className="ctx">
            <b>{DEFAULT_SCENARIO.label}</b>
            <span>{today ? `opens ${fmtDate(today)}` : DEFAULT_SCENARIO.context}</span>
          </div>
          <select value={choice} onChange={(e) => setChoice(e.target.value)} title="Which week the board opens on. The calm week is the same team, nine weeks earlier.">
            {CHOICES.map((c) => (
              <option key={c.key} value={c.key}>{c.label}</option>
            ))}
          </select>
          <span className={`pill src ${source}`} title="Where the numbers on screen come from">
            <span className="live" />
            {source === "api" ? "Live" : "Offline fixtures"}
          </span>
          <span className="pill" title="Which ladder rung produced the scores on screen">{rung === null ? "Model —" : RUNG_LABEL[rung]}</span>
        </div>

        <main>
          <div className="pghead">
            <div className="tt">
              <div className="ttl">{current.title}</div>
              <div className="sub">{current.sub}</div>
            </div>
          </div>
          {screen === "command" && <CommandCenter scenario={scenario} day={startDay} onSource={onSource} onOpen={goto} onOpenVeteran={openVeteran} />}
          {screen === "forecast" && <Forecast scenario={scenario} day={startDay} onSource={onSource} onOpen={setScreen} />}
          {screen === "map" && <Map scenario={scenario} day={startDay} onSource={onSource} />}
          {screen === "careteam" && <CareTeam scenario={scenario} date={day ?? today} onSource={onSource} onOpenVeteran={openVeteran} />}
          {screen === "veteran" && <VeteranScreen focus={focus} onOpenMessage={() => setScreen("message")} onBack={() => setScreen("command")} />}
          {screen === "message" && <MessageScreen focus={focus} onBack={() => setScreen("veteran")} />}
          {screen === "ask" && <Ask scenario={scenario} day={startDay} onGoto={goto} />}
          {screen === "library" && <Library />}
          {screen === "report" && <Report onSource={onSource} />}
        </main>
      </div>
    </div>
  );
}

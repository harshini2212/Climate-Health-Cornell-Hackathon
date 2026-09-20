import { useCallback, useEffect, useState, type ReactNode } from "react";
import { IconCalendar, IconChart, IconHome, IconList, IconMap, IconMessage, IconSearch, IconUser } from "./components/Icons";
import { getForecast, postActions, type Source } from "./lib/api";
import { DEFAULT_SCENARIO, scenarioMeta } from "./lib/config";
import { RUNG_LABEL } from "./lib/labels";
import { DEFAULT_CAPACITY } from "./lib/types";
import { CareTeam } from "./screens/CareTeam";
import { Forecast } from "./screens/Forecast";
import { Map } from "./screens/Map";
import { MessageScreen } from "./screens/Message";
import { Report } from "./screens/Report";
import { VeteranScreen } from "./screens/VeteranCard";
import { Week, type VeteranFocus } from "./screens/Week";

/**
 * The shell: a 230px sidebar with grouped nav, a sticky topbar, a page header, then the
 * page. Screens sit in the demo's order: Week board -> Forecast -> Map -> Action list ->
 * Veteran card -> Message -> Model report. Every screen opens on the same demo day.
 */

export type ScreenKey = "week" | "forecast" | "map" | "careteam" | "veteran" | "message" | "report";

interface Screen {
  key: ScreenKey;
  group: string;
  label: string;
  title: string;
  sub: string;
  icon: ReactNode;
}

const SCREENS: Screen[] = [
  { key: "week", group: "Overview", label: "Week board", icon: <IconCalendar />, title: "The week ahead", sub: "Who to reach, under the day it is due" },
  { key: "forecast", group: "Overview", label: "Forecast", icon: <IconHome />, title: "Forecast", sub: "Hazards, sites and the panel over the next seven days" },
  { key: "map", group: "Overview", label: "Map", icon: <IconMap />, title: "Map", sub: "Expected need by ZIP, VA sites, and where the hazard lands" },
  { key: "careteam", group: "Care team", label: "Action list", icon: <IconList />, title: "Today's action list", sub: "Ranked by expected harm averted, cut at the team's real capacity" },
  { key: "veteran", group: "Care team", label: "Veteran card", icon: <IconUser />, title: "Veteran card", sub: "Five needs with their uncertainty, the drivers, the plan" },
  { key: "message", group: "Care team", label: "Message", icon: <IconMessage />, title: "Verified message", sub: "VA channel, four-word phrase, and the never-pay line" },
  { key: "report", group: "Proof", label: "Model report", icon: <IconChart />, title: "Model report", sub: "How we know it works: harm averted, recovery, calibration, fairness" },
];

export default function App() {
  const [screen, setScreen] = useState<ScreenKey>("week");
  const scenario = DEFAULT_SCENARIO.key;
  const meta = scenarioMeta(scenario);
  const [source, setSource] = useState<Source>("fixture");
  const [actNow, setActNow] = useState<number | null>(null);
  const [rung, setRung] = useState<number | null>(null);
  /** Set when a day on the week ribbon is opened, so the action list starts on that date. */
  const [day, setDay] = useState<string | null>(null);
  /** Set when a queue row is clicked, so the Veteran card and Message screens have a subject. */
  const [focus, setFocus] = useState<VeteranFocus | null>(null);
  const onSource = useCallback((s: Source) => setSource(s), []);

  /** The demo day: the first date of the forecast window every screen opens on. */
  const [today, setToday] = useState<string | null>(null);
  useEffect(() => {
    getForecast(scenario, meta.day)
      .then((f) => {
        setToday(f.dates[0]);
        return postActions({ date: f.dates[0], capacity: { ...DEFAULT_CAPACITY }, scenario });
      })
      .then((r) => {
        setActNow(r.counts_by_tier.act_now ?? 0);
        setRung(r.model_rung);
      })
      .catch(() => undefined);
  }, [scenario, meta.day]);

  const openDay = useCallback((d: string) => {
    setDay(d);
    setScreen("careteam");
  }, []);
  const openVeteran = useCallback((f: VeteranFocus) => {
    setFocus(f);
    setScreen("veteran");
  }, []);

  const current = SCREENS.find((s) => s.key === screen)!;
  const groups = [...new Set(SCREENS.map((s) => s.group))];

  return (
    <div className="app">
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
          <div className="note">Synthetic people, real places. Every neighbourhood rate is public and cited; no veteran here is real.</div>
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
            <b>{meta.label}</b>
            <span>{meta.context}</span>
          </div>
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
          {screen === "week" && <Week scenario={scenario} onSource={onSource} onOpenDay={openDay} onOpenVeteran={openVeteran} />}
          {screen === "forecast" && <Forecast scenario={scenario} onSource={onSource} onOpen={setScreen} />}
          {screen === "map" && <Map scenario={scenario} onSource={onSource} />}
          {screen === "careteam" && <CareTeam scenario={scenario} date={day ?? today} onSource={onSource} onOpenVeteran={openVeteran} />}
          {screen === "veteran" && <VeteranScreen focus={focus} onOpenMessage={() => setScreen("message")} onBack={() => setScreen("week")} />}
          {screen === "message" && <MessageScreen focus={focus} onBack={() => setScreen("veteran")} />}
          {screen === "report" && <Report onSource={onSource} />}
        </main>
      </div>
    </div>
  );
}

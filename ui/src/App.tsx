import { useCallback, useEffect, useState, type ReactNode } from "react";
import { IconChart, IconHome, IconList, IconMap, IconMessage, IconSearch, IconUser } from "./components/Icons";
import { postActions, type Source } from "./lib/api";
import { RUNG_LABEL } from "./lib/labels";
import { DEFAULT_CAPACITY } from "./lib/types";
import { CareTeam } from "./screens/CareTeam";
import { Forecast } from "./screens/Forecast";
import { Map } from "./screens/Map";

/**
 * The shell, ported from the Brexify command-center template: a 230px sidebar with grouped
 * nav, a sticky topbar, a page header, then the page. Screens are registered below in the
 * demo's order: Forecast -> Map -> Care team -> Veteran card -> Message -> Model report.
 */

export type ScreenKey = "forecast" | "map" | "careteam" | "veteran" | "message" | "report";

interface Screen {
  key: ScreenKey;
  group: string;
  label: string;
  title: string;
  sub: string;
  icon: ReactNode;
}

const SCREENS: Screen[] = [
  { key: "forecast", group: "Overview", label: "Forecast", icon: <IconHome />, title: "Forecast", sub: "Command center · next 7 days" },
  { key: "map", group: "Overview", label: "Map", icon: <IconMap />, title: "Map", sub: "Expected need by ZIP, VA sites, hazards" },
  { key: "careteam", group: "Care team", label: "Action list", icon: <IconList />, title: "Today's action list", sub: "Cut at the team's real capacity" },
  { key: "veteran", group: "Care team", label: "Veteran card", icon: <IconUser />, title: "Veteran card", sub: "Five needs, drivers, plan" },
  { key: "message", group: "Care team", label: "Message", icon: <IconMessage />, title: "Verified message", sub: "VA channel, phrase, never-pay line" },
  { key: "report", group: "Proof", label: "Model report", icon: <IconChart />, title: "Model report", sub: "Recovery, calibration, harm averted, fairness" },
];

const SCENARIOS = [
  { key: "sandy_then_heat", label: "Sandy, then heat" },
  { key: "ida_flash_flood", label: "Ida flash flood" },
  { key: "smoke_2023", label: "Smoke, June 2023" },
];

function ComingSoon({ what, lane }: { what: string; lane: string }) {
  return (
    <div className="page">
      <div className="empty">
        <div className="eh">{what}</div>
        This screen is wired to the <code>{lane}</code> lane's response shape and lands when that lane merges.
      </div>
    </div>
  );
}

export default function App() {
  const [screen, setScreen] = useState<ScreenKey>("forecast");
  const [scenario, setScenario] = useState(SCENARIOS[0].key);
  const [source, setSource] = useState<Source>("fixture");
  const [actNow, setActNow] = useState<number | null>(null);
  const [rung, setRung] = useState<number | null>(null);
  const onSource = useCallback((s: Source) => setSource(s), []);

  // The nav badge on "Action list" is today's Act-now count at default capacity.
  useEffect(() => {
    postActions({ date: "", capacity: { ...DEFAULT_CAPACITY }, scenario })
      .then((r) => {
        setActNow(r.counts_by_tier.act_now ?? 0);
        setRung(r.model_rung);
      })
      .catch(() => undefined);
  }, [scenario]);

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
                <a key={s.key} className={s.key === screen ? "on" : ""} onClick={() => setScreen(s.key)} role="tab" aria-selected={s.key === screen}>
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
            <span>Search veterans, ZIPs…</span>
            <span className="kbd">⌘K</span>
          </div>
          <div className="sp" />
          <select value={scenario} onChange={(e) => setScenario(e.target.value)} title="Scenario">
            {SCENARIOS.map((s) => (
              <option key={s.key} value={s.key}>
                {s.label}
              </option>
            ))}
          </select>
          <span className={`pill src ${source}`} title="Where the numbers on screen come from">
            <span className="live" />
            {source === "api" ? "live API" : "fixtures"}
          </span>
          <span className="cost">{rung === null ? "" : RUNG_LABEL[rung]}</span>
          <span className="pillbadge">synthetic people · real places</span>
        </div>

        <main>
          <div className="pghead">
            <div className="tt">
              <div className="ttl">{current.title}</div>
              <div className="sub">{current.sub}</div>
            </div>
          </div>
          {screen === "forecast" && <Forecast scenario={scenario} onSource={onSource} onOpen={setScreen} />}
          {screen === "map" && <Map scenario={scenario} onSource={onSource} />}
          {screen === "careteam" && <CareTeam scenario={scenario} onSource={onSource} />}
          {screen === "veteran" && <ComingSoon what="Walter's card: five needs as 10–90% interval bars, drivers, medication panel, the plan" lane="api" />}
          {screen === "message" && <ComingSoon what="The message the veteran gets, with the five mandatory elements as a visible checklist" lane="api" />}
          {screen === "report" && <ComingSoon what="Recovery dot-whisker, reliability curves, harm averted vs baselines, ablations, fairness table" lane="eval" />}
        </main>
      </div>
    </div>
  );
}

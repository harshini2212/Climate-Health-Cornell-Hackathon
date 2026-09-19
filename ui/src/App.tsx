import { useCallback, useState } from "react";
import type { Source } from "./lib/api";
import { CareTeam } from "./screens/CareTeam";
import { Map } from "./screens/Map";

type Screen = "map" | "careteam";
const SCENARIO = "sandy_then_heat";

export default function App() {
  const [screen, setScreen] = useState<Screen>("map");
  const [source, setSource] = useState<Source>("fixture");
  const onSource = useCallback((s: Source) => setSource(s), []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          Leeward<small>care that gets ahead of the weather</small>
        </div>
        <nav className="tabs" role="tablist">
          <button role="tab" aria-selected={screen === "map"} onClick={() => setScreen("map")}>
            Map
          </button>
          <button role="tab" aria-selected={screen === "careteam"} onClick={() => setScreen("careteam")}>
            Care team
          </button>
        </nav>
        <div className="spacer" />
        <span className="badge" style={{ color: "var(--muted)" }}>synthetic people · real places</span>
        <span className={`badge ${source}`} title="Where the numbers on screen came from">
          <span className="dot" />
          {source === "api" ? "live API" : "fixtures"}
        </span>
      </header>
      {screen === "map" ? <Map scenario={SCENARIO} onSource={onSource} /> : <CareTeam scenario={SCENARIO} onSource={onSource} />}
    </div>
  );
}

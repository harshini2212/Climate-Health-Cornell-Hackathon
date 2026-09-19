import { useCallback, useState } from "react";
import type { Source } from "./lib/api";
import { CareTeam } from "./screens/CareTeam";
import { Map } from "./screens/Map";
import { Week } from "./screens/Week";

type Screen = "week" | "map" | "careteam";
const SCENARIO = "sandy_then_heat";

export default function App() {
  // The week board is the default: the team wants a board on a side screen, not a page
  // they remember to visit.
  const [screen, setScreen] = useState<Screen>("week");
  const [source, setSource] = useState<Source>("fixture");
  /** Set when a day on the ribbon is clicked, so the care-team view opens on that date. */
  const [day, setDay] = useState<string | null>(null);
  const onSource = useCallback((s: Source) => setSource(s), []);

  const openDay = useCallback((date: string) => {
    setDay(date);
    setScreen("careteam");
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          Leeward<small>care that gets ahead of the weather</small>
        </div>
        <nav className="tabs" role="tablist">
          <button role="tab" aria-selected={screen === "week"} onClick={() => setScreen("week")}>
            Week
          </button>
          <button role="tab" aria-selected={screen === "map"} onClick={() => setScreen("map")}>
            Map
          </button>
          <button role="tab" aria-selected={screen === "careteam"} onClick={() => setScreen("careteam")}>
            Care team{day ? ` · ${day}` : ""}
          </button>
        </nav>
        <div className="spacer" />
        <span className="badge" style={{ color: "var(--muted)" }}>synthetic people · real places</span>
        <span className={`badge ${source}`} title="Where the numbers on screen came from">
          <span className="dot" />
          {source === "api" ? "live API" : "fixtures"}
        </span>
      </header>
      {screen === "week" && <Week scenario={SCENARIO} onSource={onSource} onOpenDay={openDay} />}
      {screen === "map" && <Map scenario={SCENARIO} onSource={onSource} />}
      {screen === "careteam" && <CareTeam scenario={SCENARIO} date={day} onSource={onSource} />}
    </div>
  );
}

/**
 * Ask Leeward, full screen. The same panel the dashboard carries, with the week's data
 * behind it and the resource library beside it.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { AskPanel } from "../components/AskPanel";
import { eventDays } from "../components/EventFeed";
import { getForecast, postActions, postActionsWeek } from "../lib/api";
import type { AskContext } from "../lib/ask";
import { LIBRARY } from "../lib/library";
import { DEFAULT_CAPACITY, type ActionsResponse, type ForecastResponse } from "../lib/types";
import type { ScreenKey } from "../App";
import { ResourceCard } from "./Library";

export function Ask({ scenario, day, onGoto }: { scenario: string; day?: number; onGoto: (s: ScreenKey, d?: string) => void }) {
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [week, setWeek] = useState<(ActionsResponse | null)[]>([]);
  const [open, setOpen] = useState<string | null>(null);

  const load = useCallback(async () => {
    const fc = await getForecast(scenario, day);
    setForecast(fc);
    const dates = fc.dates.slice(0, 7);
    setWeek(new Array(dates.length).fill(null));
    const today = await postActions({ date: dates[0], capacity: DEFAULT_CAPACITY, scenario });
    setWeek((w) => w.map((x, i) => (i === 0 ? { ...today, date: dates[0] } : x)));
    await postActionsWeek(dates.slice(1), { capacity: DEFAULT_CAPACITY, scenario }, (i, r) => setWeek((w) => w.map((x, j) => (j === i + 1 ? r : x))));
  }, [scenario, day]);

  useEffect(() => {
    void load().catch(() => undefined);
  }, [load]);

  const days = useMemo(() => (forecast ? eventDays(forecast.dates.slice(0, 7), forecast.zips) : []), [forecast]);
  const ctx: AskContext = useMemo(() => ({
    days: days.map((d) => ({ date: d.date, flood: d.flood, heat: d.heat, smoke: d.smoke, surge: d.surge, outage: d.outagePeak, maxHeatIndex: d.maxHeatIndex, maxPm25: d.maxPm25 })),
    week,
    facilities: forecast?.facilities ?? [],
    panel: week[0]?.n_panel ?? 0,
  }), [days, week, forecast]);

  return (
    <div className="page row g2" style={{ alignItems: "start" }}>
      <AskPanel ctx={ctx} onGoto={(s, d) => onGoto(s as ScreenKey, d)} />
      <div className="card">
        <div className="ch"><h3>What it answers from</h3><span className="sub">the data on screen, and these</span></div>
        <div className="lib-mini">
          {LIBRARY.slice(0, 5).map((r) => <ResourceCard key={r.id} r={r} open={open === r.id} onToggle={() => setOpen(open === r.id ? null : r.id)} />)}
        </div>
        <button className="sm ghost" style={{ marginTop: 12 }} onClick={() => onGoto("library")}>All {LIBRARY.length} resources</button>
        <p className="muted small" style={{ marginBottom: 0 }}>Answers are computed from the forecast, the allocator's list and the library, with no model call and no network, so the same question gives the same answer on stage.</p>
      </div>
    </div>
  );
}

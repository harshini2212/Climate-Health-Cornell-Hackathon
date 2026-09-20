/**
 * The one place the UI talks to the outside. Every call tries the FastAPI app first
 * (proxied at /api in dev) and, if that fails, answers from the committed fixtures in
 * ui/public/fixtures/. Screens never know which one they got; they read `lastSource()`
 * only to show a small badge.
 *
 * Fixture mode is deliberately behavioural, not just shaped: the capacity cut, the dense
 * ranks and the baselines are computed here from the real allocator's candidate list, so
 * the CapacitySlider tells the same story with the network off.
 */

import type {
  ActionRow,
  ActionsRequest,
  ActionsResponse,
  BaselineResult,
  Capacity,
  ForecastResponse,
  Message,
  ReportResponse,
  ScoresResponse,
  Tier,
  VeteranCard,
} from "./types";

export type Source = "api" | "fixture";
let source: Source = "fixture";
export const lastSource = (): Source => source;

export const API = import.meta.env.VITE_API_URL ?? "/api";
const FIXTURES = `${import.meta.env.BASE_URL}fixtures`;

/**
 * A 10k-veteran allocation takes ~0.6 s live and the API serves one at a time, so a week
 * queued behind other requests can take several seconds. The proxy fails in ~1 ms when the
 * API is down, so a long timeout costs nothing offline.
 */
const TIMEOUT_MS = 30_000;

/**
 * A build with no API behind it: `VITE_STATIC=1 npm run build`. Every call goes straight
 * to the committed fixtures, so a published bundle never waits on a request that cannot
 * succeed, and the badge honestly reads "Offline fixtures".
 */
const STATIC_ONLY = import.meta.env.VITE_STATIC === "1";

async function tryApi<T>(path: string, init?: RequestInit): Promise<T | null> {
  if (STATIC_ONLY) return null;
  try {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
    const res = await fetch(`${API}${path}`, { ...init, signal: ctl.signal });
    clearTimeout(timer);
    if (!res.ok) return null;
    source = "api";
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

const fixtureCache = new Map<string, Promise<unknown>>();
function fixture<T>(name: string): Promise<T> {
  if (!fixtureCache.has(name)) {
    fixtureCache.set(
      name,
      fetch(`${FIXTURES}/${name}.json`).then((r) => {
        if (!r.ok) throw new Error(`fixture ${name}.json missing; run scripts/make_ui_fixtures.py`);
        return r.json();
      }),
    );
  }
  source = "fixture";
  return fixtureCache.get(name) as Promise<T>;
}

// ---------------------------------------------------------------------------

const forecastCache = new Map<string, Promise<ForecastResponse>>();

/**
 * `day` omitted asks the API for the window the demo opens on (leeward/demo.py: two days
 * in front of the first alert), rather than day 0, which is nine weeks before anything
 * happens. Offline there is one forecast fixture, built for the landfall week.
 */
export function getForecast(scenario: string, day?: number): Promise<ForecastResponse> {
  const key = `${scenario}|${day ?? "opening"}`;
  if (!forecastCache.has(key)) {
    forecastCache.set(key, (async () => {
      const q = day === undefined ? "" : `&day=${day}`;
      const live = await tryApi<ForecastResponse>(`/forecast?scenario=${scenario}${q}`);
      return live ?? fixture<ForecastResponse>("forecast");
    })());
  }
  return forecastCache.get(key)!;
}

/** Fixture shape: { [date]: { [need]: ScoresResponse } }. */
type ScoresFixture = Record<string, Record<string, ScoresResponse>>;

export async function getScores(date: string, need: string): Promise<ScoresResponse> {
  const live = await tryApi<ScoresResponse>(`/scores?date=${date}&need=${need}`);
  if (live) return live;
  const all = await fixture<ScoresFixture>("scores");
  const byNeed = all[date] ?? all[Object.keys(all)[0]];
  return byNeed[need];
}

// ---------------------------------------------------------------------------
// Actions. The fixture is a *candidate list*; the capacity cut happens here.

interface Candidate extends ActionRow {
  age: number;
  n_chronic: number;
  rand: number;
}

interface ActionsFixture {
  date: string;
  model_rung: number;
  n_panel: number;
  candidates: Candidate[];
}

function cut(cands: Candidate[], capacity: Capacity, order: (c: Candidate) => number): Candidate[] {
  const sorted = [...cands].sort((a, b) => order(b) - order(a));
  const left: Capacity = { ...capacity };
  const perVet = new Map<string, number>();
  const kept: Candidate[] = [];
  for (const c of sorted) {
    const limit = c.tier === ("act_now" as Tier) ? 3 : 1;
    if ((perVet.get(c.veteran_id) ?? 0) >= limit) continue;
    if ((left[c.capacity_bucket] ?? 0) <= 0) continue;
    left[c.capacity_bucket] -= 1;
    perVet.set(c.veteran_id, (perVet.get(c.veteran_id) ?? 0) + 1);
    kept.push(c);
  }
  return kept;
}

const sum = (rows: { eha: number }[]) => rows.reduce((s, r) => s + r.eha, 0);

export function allocateFixture(fx: ActionsFixture, capacity: Capacity): ActionsResponse {
  const chosen = cut(fx.candidates, capacity, (c) => c.eha);
  const actions: ActionRow[] = chosen
    .sort((a, b) => b.eha - a.eha)
    .map(({ age: _a, n_chronic: _n, rand: _r, ...row }, i) => ({ ...row, rank: i + 1 }));
  const counts: Record<string, number> = {};
  for (const a of actions) counts[a.tier] = (counts[a.tier] ?? 0) + 1;
  const baselines: BaselineResult[] = [
    { name: "leeward", total_eha: sum(actions) },
    { name: "rank_by_age", total_eha: sum(cut(fx.candidates, capacity, (c) => c.age)) },
    { name: "rank_by_chronic", total_eha: sum(cut(fx.candidates, capacity, (c) => c.n_chronic)) },
    { name: "random", total_eha: sum(cut(fx.candidates, capacity, (c) => c.rand)) },
  ];
  return {
    date: fx.date,
    capacity,
    actions,
    total_eha: sum(actions),
    baselines,
    n_panel: fx.n_panel,
    n_selected: actions.length,
    counts_by_tier: counts,
    model_rung: fx.model_rung,
  };
}

/** Responses are deterministic for a given request, so a click never re-runs an allocation. */
const actionsCache = new Map<string, Promise<ActionsResponse>>();

export function postActions(req: ActionsRequest): Promise<ActionsResponse> {
  const key = JSON.stringify(req);
  if (!actionsCache.has(key)) {
    actionsCache.set(key, (async () => {
      const live = req.date
        ? await tryApi<ActionsResponse>("/actions", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(req) })
        : null;
      if (live) return live;
      const fx = await fixture<ActionsFixture>("actions_candidates");
      return allocateFixture(fx, req.capacity);
    })());
  }
  return actionsCache.get(key)!;
}

/**
 * The week, a day at a time. ActionsRequest is single-day and the api lane owns that
 * shape, so a week is seven requests; the API serves them one at a time, so they go out
 * two at a time and `onDay` lets the board fill as each lands rather than all at once.
 */
export async function postActionsWeek(
  dates: string[],
  base: Omit<ActionsRequest, "date">,
  onDay?: (index: number, resp: ActionsResponse) => void,
  concurrency = 2,
): Promise<ActionsResponse[]> {
  const out: ActionsResponse[] = new Array(dates.length);
  let next = 0;
  const worker = async () => {
    while (next < dates.length) {
      const i = next++;
      const r = await postActions({ ...base, date: dates[i] });
      const fixed = r.date === dates[i] ? r : { ...r, date: dates[i] };
      out[i] = fixed;
      onDay?.(i, fixed);
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, dates.length) }, worker));
  return out;
}

// ---------------------------------------------------------------------------
// Detail views and the report. Fixtures are keyed by id, so a click is a map lookup.

export async function getVeteran(veteranId: string, date: string): Promise<VeteranCard> {
  const live = await tryApi<VeteranCard>(`/veteran/${veteranId}?date=${date}`);
  if (live) return live;
  const all = await fixture<Record<string, VeteranCard>>("veterans");
  const card = all[veteranId];
  if (!card) throw new Error(`no veteran card for ${veteranId}`);
  return card;
}

export async function getMessage(actionId: string): Promise<Message> {
  const live = await tryApi<Message>(`/message/${actionId}`);
  if (live) return live;
  const all = await fixture<Record<string, Message>>("messages");
  const msg = all[actionId];
  if (!msg) throw new Error(`no message for action ${actionId}`);
  return msg;
}

/**
 * `report/report.json` is generated output and gitignored, so a clean clone answers this
 * route 200 with the rung and nothing else -- which means "make report has not been run
 * here", not "the audit passed". On screen the two are indistinguishable, so an empty live
 * report falls through to the committed fixture and the badge says `fixture`. Report.tsx
 * prints `generated_at` either way, so whichever one you are looking at says when it ran.
 */
export async function getReport(): Promise<ReportResponse> {
  const live = await tryApi<ReportResponse>("/report");
  if (live && (live.recovery.length || live.calibration.length || live.fairness.length)) return live;
  return fixture<ReportResponse>("report");
}

/** The partner-sheet CSV only exists live; offline the button explains why. */
export const exportUrl = (date: string) => `${API}/export?date=${date}`;

/**
 * The one place the UI talks to the outside. Every call tries the FastAPI app first
 * (proxied at /api in dev) and, if that fails, answers from the committed fixtures in
 * ui/public/fixtures/. Screens never know which one they got; they read `lastSource()`
 * only to show a small badge.
 *
 * Fixture mode is deliberately behavioural, not just shaped: the capacity cut, the dense
 * ranks and the baselines are computed here from a candidate list, so the CapacitySlider
 * tells the same story before and after the real allocator lands.
 */

import type {
  ActionRow,
  ActionsRequest,
  ActionsResponse,
  BaselineResult,
  Capacity,
  ForecastResponse,
  Message,
  ScoresResponse,
  Tier,
  VeteranCard,
} from "./types";

export type Source = "api" | "fixture";
let source: Source = "fixture";
export const lastSource = (): Source => source;

const API = import.meta.env.VITE_API_URL ?? "/api";
const FIXTURES = `${import.meta.env.BASE_URL}fixtures`;

async function tryApi<T>(path: string, init?: RequestInit): Promise<T | null> {
  try {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 1500);
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

/**
 * `day` omitted asks the API for the window the demo opens on (leeward/demo.py), rather
 * than day 0, which in this scenario is nine weeks before anything happens. Offline there
 * is one forecast fixture and it is already built for that window, so both agree.
 */
export async function getForecast(scenario: string, day?: number): Promise<ForecastResponse> {
  const q = day === undefined ? "" : `&day=${day}`;
  const live = await tryApi<ForecastResponse>(`/forecast?scenario=${scenario}${q}`);
  return live ?? fixture<ForecastResponse>("forecast");
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
  // Ranks are dense and ordered by EHA, exactly what test_guardrails asserts of the real one.
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

export async function postActions(req: ActionsRequest): Promise<ActionsResponse> {
  const live = await tryApi<ActionsResponse>("/actions", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(req),
  });
  if (live) return live;
  const fx = await fixture<ActionsFixture>("actions_candidates");
  return allocateFixture(fx, req.capacity);
}

// ---------------------------------------------------------------------------
// The week. ActionsRequest is single-day and the api lane owns that shape, so a
// week is seven requests in parallel rather than a new field. Offline each one
// falls through to allocateFixture, so the board fills with the network off.

export async function postActionsWeek(
  dates: string[],
  base: Omit<ActionsRequest, "date">,
): Promise<ActionsResponse[]> {
  const week = await Promise.all(dates.map((date) => postActions({ ...base, date })));
  // Fixture mode answers from one modelled day and stamps it with that day's date.
  // The board is keyed by the date it asked for, so put that back.
  return week.map((r, i) => (r.date === dates[i] ? r : { ...r, date: dates[i] }));
}

// ---------------------------------------------------------------------------
// The two detail views. Fixtures are keyed by id, so a click is a map lookup.

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

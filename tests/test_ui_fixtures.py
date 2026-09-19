"""The UI fixtures are typed against leeward/api/schemas.py, and this is what enforces it.

If a fixture stops parsing through the pydantic models, the UI has been built against a
shape the API will never send. Also checks that the client-side capacity cut the UI does
in fixture mode obeys the same rules the real allocator must obey.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leeward.api import schemas as api
from leeward.schema import DEFAULT_CAPACITY, NEEDS

FIXTURES = Path(__file__).resolve().parents[1] / "ui" / "public" / "fixtures"


def _load(name: str):
    path = FIXTURES / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} missing -- run scripts/make_ui_fixtures.py")
    return json.loads(path.read_text(encoding="utf-8"))


def test_forecast_fixture_is_a_forecast_response() -> None:
    fx = api.ForecastResponse(**_load("forecast"))
    assert len(fx.dates) >= 1
    per_day = {}
    for z in fx.zips:
        per_day[z.date] = per_day.get(z.date, 0) + 1
    assert set(per_day) == set(fx.dates)
    assert min(per_day.values()) == 178 == max(per_day.values()), "every MODZCTA every day"
    assert len(fx.facilities) == 14
    assert any(f.facility_id == "630" for f in fx.facilities)


def test_scores_fixture_is_scores_responses_keyed_by_date_and_need() -> None:
    fx = _load("scores")
    assert fx, "scores fixture is empty"
    for date, by_need in fx.items():
        assert set(by_need) == set(NEEDS), f"{date}: needs {sorted(by_need)}"
        for need, payload in by_need.items():
            resp = api.ScoresResponse(**payload)
            assert resp.need == need and str(resp.date) == date
            assert len(resp.zips) > 100
            assert all(z.lo80 <= z.expected_count <= z.hi80 for z in resp.zips)


def test_candidates_are_legal_action_rows_and_cover_the_slider_range() -> None:
    fx = _load("actions_candidates")
    rows = fx["candidates"]
    keys = set(api.ActionRow.model_fields)
    for row in rows:
        api.ActionRow(**{k: row[k] for k in keys})
        assert {"age", "n_chronic", "rand"} <= set(row), "baselines need age, n_chronic, rand"
    ids = [r["action_id"] for r in rows]
    assert len(ids) == len(set(ids)), "action_id must be unique; React keys and the log depend on it"
    calls = sum(1 for r in rows if r["capacity_bucket"] == "call")
    assert calls >= 40, f"only {calls} call candidates; the default capacity is 40"
    assert fx["model_rung"] in (0, 1, 2, 3)


def _cut(rows, capacity, key):
    left = dict(capacity)
    per_vet: dict[str, int] = {}
    kept = []
    for r in sorted(rows, key=key, reverse=True):
        limit = 3 if r["tier"] == "act_now" else 1
        if per_vet.get(r["veteran_id"], 0) >= limit or left.get(r["capacity_bucket"], 0) <= 0:
            continue
        left[r["capacity_bucket"]] -= 1
        per_vet[r["veteran_id"]] = per_vet.get(r["veteran_id"], 0) + 1
        kept.append(r)
    return kept


def test_fixture_mode_cut_obeys_capacity_and_is_monotone() -> None:
    """Mirrors allocateFixture() in ui/src/lib/api.ts. Same rules as test_guardrails."""
    rows = _load("actions_candidates")["candidates"]
    prev = None
    for calls in (10, 20, 40, 80, 100):
        cap = dict(DEFAULT_CAPACITY, call=calls)
        kept = _cut(rows, cap, key=lambda r: r["eha"])
        used: dict[str, int] = {}
        for r in kept:
            used[r["capacity_bucket"]] = used.get(r["capacity_bucket"], 0) + 1
        assert all(n <= cap[b] for b, n in used.items()), f"capacity exceeded at {calls} calls"
        total = sum(r["eha"] for r in kept)
        if prev is not None:
            assert total >= prev - 1e-9, "more capacity averted less harm"
        prev = total
        resp = api.ActionsResponse(
            date=rows[0] and _load("actions_candidates")["date"], capacity=cap,
            actions=[api.ActionRow(**{**{k: r[k] for k in api.ActionRow.model_fields}, "rank": i + 1})
                     for i, r in enumerate(sorted(kept, key=lambda r: r["eha"], reverse=True))],
            total_eha=total, n_panel=1, n_selected=len(kept), model_rung=0)
        assert resp.n_selected == len(kept)

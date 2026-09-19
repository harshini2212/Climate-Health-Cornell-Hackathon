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
from leeward.schema import CHANNELS, DEFAULT_CAPACITY, MANDATORY_MESSAGE_ELEMENTS, NEEDS

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "ui" / "public" / "fixtures"


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
    assert calls >= 100, f"only {calls} call candidates; the slider goes to 100"
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


# --------------------------------------------------------------------------- #
# The week board's detail views. Week.tsx opens VeteranCard.tsx and Message.tsx
# from these two fixtures when the network is off, so they carry the same
# promises the live routes will have to carry.
# --------------------------------------------------------------------------- #

def test_veteran_fixture_is_veteran_cards_keyed_by_id() -> None:
    fx = _load("veterans")
    cands = _load("actions_candidates")
    assert set(c["veteran_id"] for c in cands["candidates"]) <= set(fx), (
        "a candidate has no veteran card; clicking that queue card would open nothing")
    for vid, payload in fx.items():
        card = api.VeteranCard(**payload)
        assert card.veteran_id == vid
        assert {n.need for n in card.needs} == set(NEEDS), f"{vid}: needs {card.needs}"
        for n in card.needs:
            assert n.p_lo80 <= n.p_mean <= n.p_hi80, f"{vid}/{n.need}: interval excludes the mean"
            assert 0.0 <= n.p_epistemic_share <= 1.0
        assert card.why_this_tier, "every card says why it landed in its tier"
        assert card.is_synthetic, "these are synthetic people and the card must say so"


def test_message_fixture_carries_every_mandatory_element() -> None:
    """The same promise test_guardrails makes of leeward.outreach.messages, made here of
    the text the UI actually renders offline. A veteran must be able to tell this from a scam."""
    fx = _load("messages")
    cands = _load("actions_candidates")
    assert set(c["action_id"] for c in cands["candidates"]) <= set(fx), (
        "a candidate action has no message; the Message panel would open empty")
    for aid, payload in fx.items():
        msg = api.Message(**payload)
        assert msg.action_id == aid
        for element in MANDATORY_MESSAGE_ELEMENTS:
            assert element in msg.body, f"message {aid} is missing {element!r}"
        assert "Veterans Crisis Line" in msg.body, f"message {aid} omits the crisis line"
        assert len(msg.verification_phrase.split()) == 4, "the verification phrase is four words"
        assert msg.verification_phrase in msg.body, "the phrase must appear in the text we send"
        assert msg.channel in CHANNELS, f"{aid}: channel {msg.channel!r}"
        assert msg.includes_never_pay_line and msg.includes_vsafe and msg.includes_crisis_line
    for bad in ("bit.ly", "tinyurl", "t.co/", "goo.gl"):
        assert not any(bad in m["body"] for m in fx.values()), f"shortener {bad!r} in outreach"


def test_no_queue_card_line_names_a_diagnosis() -> None:
    """Week.tsx shows `rationale` on the de-identified queue card and keeps `top_driver`
    behind the reveal, because driver phrases name conditions and medicines by design."""
    words = ("copd", "asthma", "ptsd", "dialysis", "insulin", "cancer", "depression",
             "diabetes", "inhaler", "therapy", "methadone")
    for row in _load("actions_candidates")["candidates"]:
        low = row["rationale"].lower()
        assert not any(w in low for w in words), (
            f"rationale for {row['action_id']} names a condition: {row['rationale']!r}. "
            "It is rendered on a wall-mounted board next to a de-identified handle.")


def test_week_board_shows_no_name_no_diagnosis_and_no_bare_eha() -> None:
    """The board hangs on a wall in a shared clinical space.

    A screenshot proves this once; this proves it every time anyone edits the screen.
    The handle is initials plus the last four of the id, so `name_display` may only reach
    `handleFor`, and the clinical fields -- driver phrases, conditions, medicines -- stay
    in VeteranCard.tsx behind a click.
    """
    src = (ROOT / "ui" / "src" / "screens" / "Week.tsx").read_text(encoding="utf-8")
    body = src.split("export function Week(", 1)[1]

    for banned, why in [
        ("top_driver", "driver phrases name conditions and medicines"),
        ("conditions", "the condition list belongs behind the reveal"),
        (".eha", "never show a bare expected-harm number as the reason"),
        ("medications", "the medication panel belongs behind the reveal"),
    ]:
        assert banned not in body, f"Week.tsx renders {banned!r}: {why}"

    uses = [ln.strip() for ln in src.splitlines() if "name_display" in ln]
    assert uses, "Week.tsx no longer builds a handle at all"
    for line in uses:
        assert "handleFor(" in line or "nameDisplay" in line, (
            f"Week.tsx touches name_display outside handleFor: {line!r}")


def test_handle_never_leaks_more_than_initials() -> None:
    """handleFor('James Okafor', 'SYN-000309') -> 'J.O. - 0309' and nothing more."""
    src = (ROOT / "ui" / "src" / "screens" / "Week.tsx").read_text(encoding="utf-8")
    fn = src.split("export function handleFor(", 1)[1].split("\n}", 1)[0]
    assert "slice(0, 2)" in fn, "the handle takes at most two initials"
    assert "[0]" in fn, "the handle takes the first letter of a name part, never the part"
    assert "slice(-4)" in fn, "the handle takes the last four of the id, never the whole id"

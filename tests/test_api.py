"""The API: eight routes, the frozen shapes, and the promises the demo makes on stage.

Every test runs against its own data directory, built here by the fixture generator and the
real allocator, so nothing depends on what is in `data/` and `POST /log` never touches it.
The routes are held to what the decision layer and the tables say, not to their own output:
totals are recomputed from `scores.parquet`, plans from `allocate()`, tiers from `tiers.assign`.
"""

from __future__ import annotations

import importlib.util
import os
import re
import statistics
import sys
import time
import types
from collections import Counter
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from leeward import schema
from leeward.api import main as api_main
from leeward.api import schemas as api
from leeward.api import store
from leeward.decision import severity, tiers
from leeward.decision.allocate import allocate, total_eha
from leeward.outreach.export import CONSENT_FOR
from leeward.schema import ACTION_COST_UNIT, DEFAULT_CAPACITY, MANDATORY_MESSAGE_ELEMENTS, NEEDS

ROOT = Path(__file__).resolve().parents[1]
DAY = date(2026, 7, 16)           # landfall in the fixtures: flood warnings, station 630 closed


# --------------------------------------------------------------------------- #
# An isolated data directory: fixtures, then the real allocator's plan for every day
# --------------------------------------------------------------------------- #

def _build_data(data: Path, mp: pytest.MonkeyPatch) -> None:
    mp.setattr(schema, "DATA", data)
    mp.setattr(store, "REPORT_DIR", data.parent / "report")
    spec = importlib.util.spec_from_file_location("make_fixtures",
                                                  ROOT / "scripts" / "make_fixtures.py")
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    with pytest.MonkeyPatch.context() as local:
        local.setattr(sys, "argv", ["make_fixtures.py"])     # the generator parses argv
        assert fixtures.main() == 0
    plan = allocate(schema.read("scores"), schema.read("cohort"), DEFAULT_CAPACITY)
    schema.write(plan, "actions")


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("leeward")
    _build_data(root / "data", mp)
    yield root / "data"
    mp.undo()


@pytest.fixture(scope="module")
def client(data: Path) -> TestClient:
    with TestClient(api_main.app) as c:
        yield c


def _post(client: TestClient, **body) -> dict:
    r = client.post("/actions", json={"date": str(DAY), **body})
    assert r.status_code == 200, r.text
    return r.json()


def _table(name: str) -> pl.DataFrame:
    return schema.read(name)


# --------------------------------------------------------------------------- #
# The surface
# --------------------------------------------------------------------------- #

def test_all_eight_routes_exist(client: TestClient) -> None:
    routes = {(m, r.path) for r in api_main.app.routes for m in getattr(r, "methods", ())}
    assert {("GET", "/forecast"), ("GET", "/scores"), ("GET", "/veteran/{veteran_id}"),
            ("POST", "/actions"), ("GET", "/message/{action_id}"), ("POST", "/log"),
            ("GET", "/report"), ("GET", "/export")} <= routes


def test_the_api_never_imports_the_model() -> None:
    """Nothing runs inference in a request. The cheapest way to keep that true is to make it
    impossible to reach: the API package does not import the model, NumPyro or JAX."""
    for path in (ROOT / "leeward" / "api").glob("*.py"):
        src = path.read_text()
        for banned in ("numpyro", "import jax", "from jax", "arviz", "leeward.model"):
            assert banned not in src, f"{path.name} mentions {banned!r}"


# --------------------------------------------------------------------------- #
# The built UI, served by the API: the demo is one process
# --------------------------------------------------------------------------- #

def test_the_built_ui_is_served_at_the_root(client: TestClient) -> None:
    """`ui/dist` is committed so a clone with no npm and no wifi can still show a page.

    Fetching the page is not enough -- a page whose script 404s is a white screen -- so every
    asset `index.html` points at has to come back too, and as itself rather than as HTML.
    """
    assert (api_main.UI_DIST / "index.html").is_file(), "ui/dist is not built: run `make ui`"
    page = client.get("/")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert 'id="root"' in page.text
    refs = re.findall(r'(?:src|href)="(/assets/[^"]+)"', page.text)
    assert any(r.endswith(".js") for r in refs), "index.html loads no script"
    for ref in refs:
        got = client.get(ref)
        assert got.status_code == 200 and "text/html" not in got.headers["content-type"], ref


def test_the_ui_calls_the_api_under_the_prefix_vite_strips_in_dev(client: TestClient) -> None:
    """The UI asks for `/api/forecast`. The Vite proxy strips `/api`; served from here, this
    app has to, or a working bundle quietly falls back to fixtures and nobody sees it."""
    query = {"scenario": "sandy_then_heat", "day": 5}
    plain, prefixed = client.get("/forecast", params=query), client.get("/api/forecast", params=query)
    assert prefixed.status_code == 200 and prefixed.content == plain.content
    assert client.get("/api/report").json() == client.get("/report").json()
    body = {"date": str(DAY), "capacity": {"call": 3}}
    assert client.post("/api/actions", json=body).json() == client.post("/actions", json=body).json()


def test_routes_win_over_the_ui_mount_and_unknown_paths_are_not_the_app(
        client: TestClient) -> None:
    """A mount at `/` matches everything, so it is only safe if it is declared last."""
    assert client.get("/openapi.json").headers["content-type"] == "application/json"
    assert client.get("/report").headers["content-type"] == "application/json"
    assert client.get("/no-such-file.js").status_code == 404
    assert client.get("/api/no-such-route").status_code == 404


def test_mount_ui_serves_a_directory_only_if_it_has_an_index(tmp_path: Path) -> None:
    app = FastAPI()

    @app.get("/ping")
    def ping() -> dict:
        return {"ok": True}

    assert api_main.mount_ui(app, tmp_path / "missing") is False
    assert TestClient(app).get("/").status_code == 404, "an unbuilt UI must not half-mount"
    (tmp_path / "index.html").write_text('<div id="root"></div>')
    assert api_main.mount_ui(app, tmp_path) is True
    got = TestClient(app)
    assert 'id="root"' in got.get("/").text
    assert got.get("/ping").json() == {"ok": True}, "the mount shadowed a route"


# --------------------------------------------------------------------------- #
# GET /forecast
# --------------------------------------------------------------------------- #

def test_forecast_is_seven_days_of_every_zip_and_every_facility(client: TestClient) -> None:
    r = client.get("/forecast", params={"scenario": "sandy_then_heat", "day": 5})
    assert r.status_code == 200
    fc = api.ForecastResponse.model_validate(r.json())
    hazard_days = sorted(_table("hazards")["date"].unique().to_list())
    assert fc.day == 5 and fc.dates == hazard_days[5:12]
    assert len(fc.zips) == 7 * 178
    assert {z.date for z in fc.zips} == set(fc.dates)
    assert len(fc.facilities) == 14


def test_forecast_says_when_a_site_is_down(client: TestClient) -> None:
    landfall = sorted(_table("hazards")["date"].unique().to_list()).index(DAY)
    fc = api.ForecastResponse.model_validate(
        client.get("/forecast", params={"day": landfall}).json())
    down = {f.facility_id for f in fc.facilities if f.site_down}
    truth = _table("site_status").filter((pl.col("date") >= DAY) & (pl.col("site_down")))
    assert "630" in down and down <= set(truth["facility_id"].to_list())
    assert "Site down" in fc.headline
    quiet = api.ForecastResponse.model_validate(client.get("/forecast", params={"day": 0}).json())
    assert not any(f.site_down for f in quiet.facilities)


def test_forecast_rejects_what_it_cannot_serve(client: TestClient) -> None:
    assert client.get("/forecast", params={"day": 30}).status_code == 404
    assert client.get("/forecast", params={"day": -1}).status_code == 422
    r = client.get("/forecast", params={"scenario": "ida_flash_flood"})
    assert r.status_code == 422 and "sandy_then_heat" in r.json()["detail"]


def test_the_last_window_is_short_not_an_error(client: TestClient) -> None:
    fc = api.ForecastResponse.model_validate(client.get("/forecast", params={"day": 28}).json())
    assert len(fc.dates) == 2


# --------------------------------------------------------------------------- #
# GET /scores
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("need", NEEDS)
def test_scores_add_up_to_the_panel(client: TestClient, need: str) -> None:
    r = client.get("/scores", params={"date": str(DAY), "need": need})
    assert r.status_code == 200
    sc = api.ScoresResponse.model_validate(r.json())
    truth = _table("scores").filter((pl.col("date") == DAY) & (pl.col("need") == need))
    assert sc.model_rung == truth["model_rung"][0]
    assert sum(z.n_panel for z in sc.zips) == truth.height
    assert sum(z.n_panel for z in sc.facilities) == truth.height
    for rows in (sc.zips, sc.facilities):
        assert sum(z.expected_count for z in rows) == pytest.approx(truth["p_mean"].sum(),
                                                                    abs=0.001 * len(rows))
        assert all(z.lo80 <= z.expected_count <= z.hi80 for z in rows)
    assert {f.modzcta for f in sc.facilities} <= set(_table("cohort")["facility_id"].to_list())


def test_scores_reject_what_they_cannot_answer(client: TestClient) -> None:
    assert client.get("/scores", params={"date": str(DAY), "need": "boredom"}).status_code == 422
    r = client.get("/scores", params={"date": "2031-01-01", "need": "heat"})
    assert r.status_code == 404 and "2026-07-01" in r.json()["detail"]
    assert client.get("/scores", params={"need": "heat"}).status_code == 422


# --------------------------------------------------------------------------- #
# GET /veteran/{id}
# --------------------------------------------------------------------------- #

def _card(client: TestClient, vid: str, day: date = DAY) -> api.VeteranCard:
    r = client.get(f"/veteran/{vid}", params={"date": str(day)})
    assert r.status_code == 200, r.text
    return api.VeteranCard.model_validate(r.json())


def test_the_card_is_the_scores_and_the_cohort_row(client: TestClient) -> None:
    # A lead_days of zero, because the card is a statement about the veteran on DAY and the
    # plan for DAY also carries work for risk days after it, where the tier may differ.
    plan = _table("actions").filter((pl.col("date") == DAY) & (pl.col("lead_days") == 0))
    vid = plan.filter(pl.col("tier") == "act_now")["veteran_id"][0]
    card = _card(client, vid)
    vet = _table("cohort").filter(pl.col("veteran_id") == vid).row(0, named=True)
    mine = _table("scores").filter((pl.col("veteran_id") == vid) & (pl.col("date") == DAY))

    assert [n.need for n in card.needs] == NEEDS
    for n in card.needs:
        row = mine.filter(pl.col("need") == n.need).row(0, named=True)
        assert (n.p_mean, n.p_lo80, n.p_hi80, n.p_epistemic_share) == (
            row["p_mean"], row["p_lo80"], row["p_hi80"], row["p_epistemic_share"])
        assert n.drivers == [row[f"driver_{i}"] for i in (1, 2, 3) if row[f"driver_{i}"]]
        assert len(n.driver_contribs) == len(n.drivers)
    assert card.tier == tiers.assign(mine, severity.load())["tier"][0] == "act_now"
    assert (card.name_display, card.age, card.modzcta, card.borough, card.facility_id) == (
        vet["name_display"], vet["age"], vet["modzcta"], vet["borough"], vet["facility_id"])
    med = card.medications
    assert (med.n_active_meds, med.acb_score, med.cold_chain, med.controlled,
            med.mail_order_pharmacy, med.days_supply_remaining) == (
        vet["n_active_meds"], vet["acb_score"], vet["med_cold_chain"], vet["med_controlled"],
        vet["mail_order_pharmacy"], vet["days_supply_remaining"])
    assert (card.powered_equipment, card.caregiver, card.floor, card.evac_zone) == (
        vet["powered_equipment"], vet["caregiver"], vet["floor"], vet["evac_zone"])
    assert card.is_synthetic is True
    assert card.facility_name not in ("", card.facility_id)


def test_the_card_plans_what_the_cached_plan_says(client: TestClient) -> None:
    plan = _table("actions").filter(pl.col("date") == DAY)
    vid = plan.filter(pl.col("tier") == "act_now")["veteran_id"][0]
    mine = plan.filter(pl.col("veteran_id") == vid).sort("rank")
    assert 1 <= mine.height <= 3
    assert _card(client, vid).planned_actions == mine["rationale"].to_list()


def test_every_tier_explains_itself_from_the_veterans_own_numbers(client: TestClient) -> None:
    day_scores = _table("scores").filter(pl.col("date") == DAY)
    by_tier = tiers.assign(day_scores).group_by("tier").agg(pl.col("veteran_id").head(4))
    seen = {}
    for tier, vids in by_tier.iter_rows():
        for vid in vids:
            card = _card(client, vid)
            assert card.tier == tier and card.why_this_tier.strip().endswith(".")
            assert "%" in card.why_this_tier
            seen[tier] = card.why_this_tier
    assert {"act_now", "find_out", "self_serve"} <= set(seen), sorted(seen)
    assert "Act-now line" in seen["act_now"]
    assert "check-in" in seen["find_out"]
    assert "short of Act-now and Find-out" in seen["self_serve"]


def test_the_card_rejects_what_it_cannot_show(client: TestClient) -> None:
    assert client.get("/veteran/NOBODY", params={"date": str(DAY)}).status_code == 404
    assert client.get("/veteran/SYN-000000", params={"date": "2031-01-01"}).status_code == 404
    assert client.get("/veteran/SYN-000000").status_code == 422


def test_medication_notes_tie_the_prescription_list_to_todays_hazard() -> None:
    quiet = dict(med_combo_raas_diuretic=False, med_renal_triple=False, med_cold_chain=False,
                 mail_order_pharmacy=False, med_controlled=False, acb_score=0,
                 days_supply_remaining=20, va_drug_classes=[])
    hot = dict(hot_day=True, heat_index_max_f=96.4, outage_frac=0.0,
               mail_delivery_disrupted=False)
    notes = api_main._medication_notes

    assert notes(quiet, hot, False) == [], "nothing to say when nothing applies"
    combo = dict(quiet, med_combo_raas_diuretic=True)
    assert "96°F heat-index day" in notes(combo, hot, False)[0]
    assert "CDC" in notes(combo, dict(hot, hot_day=False), False)[0]
    assert "heat-index" not in notes(combo, dict(hot, hot_day=False), False)[0]
    assert "CDC" in notes(combo, None, False)[0], "no hazard row, still the combination"

    cold = dict(quiet, med_cold_chain=True)
    assert notes(cold, hot, False) == [], "refrigeration matters only if the power is out"
    assert "80%" in notes(cold, dict(hot, outage_frac=0.8), False)[0]

    mail = dict(quiet, mail_order_pharmacy=True)
    assert notes(mail, hot, False) == []
    assert "20 days of supply" in notes(mail, dict(hot, mail_delivery_disrupted=True), False)[0]

    ctl = dict(quiet, med_controlled=True)
    assert notes(ctl, hot, False) == []
    assert "retail emergency refill" in notes(ctl, hot, True)[0]

    assert "Anticholinergic burden 4" in notes(dict(quiet, acb_score=4), hot, False)[0]
    assert notes(dict(quiet, acb_score=2), hot, False) == []


def test_class_notes_use_the_va_formulary_names_from_the_reference_table() -> None:
    classes = store.med_classes()
    heat = classes.filter(pl.col("hazard") == "heat")
    quiet = dict(med_combo_raas_diuretic=False, med_renal_triple=False, med_cold_chain=False,
                 mail_order_pharmacy=False, med_controlled=False, acb_score=0,
                 days_supply_remaining=0)
    one = api_main._medication_notes(dict(quiet, va_drug_classes=["CV702"]), None, False)
    row = heat.filter(pl.col("va_class_id") == "CV702").row(0, named=True)
    assert one == [f"{row['va_class_name']} (CV702): {row['mechanism'].replace('_', ' ')}"]

    many = heat["va_class_id"].head(6).to_list()
    got = api_main._medication_notes(dict(quiet, va_drug_classes=many), None, False)
    assert len(got) == 2, "the two heaviest classes, not a wall of text"
    heaviest = (heat.filter(pl.col("va_class_id").is_in(many))
                    .sort(["weight", "va_class_id"], descending=[True, False]).head(2))
    assert got == [f"{r['va_class_name']} ({r['va_class_id']}): {r['mechanism'].replace('_', ' ')}"
                   for r in heaviest.to_dicts()]

    outage_only = classes.filter(pl.col("hazard") != "heat")["va_class_id"].to_list()
    assert api_main._medication_notes(dict(quiet, va_drug_classes=outage_only), None, False) == []


# --------------------------------------------------------------------------- #
# POST /actions: the capacity slider
# --------------------------------------------------------------------------- #

def test_actions_are_the_allocators_answer(client: TestClient) -> None:
    body = _post(client)
    resp = api.ActionsResponse.model_validate(body)
    want = allocate(_table("scores"), _table("cohort"), DEFAULT_CAPACITY, date=DAY)

    assert [a.action_id for a in resp.actions] == want.sort("rank")["action_id"].to_list()
    assert [a.eha for a in resp.actions] == want.sort("rank")["eha"].to_list()
    assert resp.total_eha == pytest.approx(total_eha(want))
    assert resp.n_selected == len(resp.actions) == want.height
    assert resp.n_panel == _table("cohort").height
    assert resp.capacity == DEFAULT_CAPACITY
    assert resp.model_rung == _table("scores")["model_rung"][0]
    assert resp.counts_by_tier == {t: want.filter(pl.col("tier") == t).height
                                   for t in schema.TIERS}


def test_actions_carry_who_and_where(client: TestClient) -> None:
    cohort = _table("cohort")
    for a in api.ActionsResponse.model_validate(_post(client)).actions[:25]:
        vet = cohort.filter(pl.col("veteran_id") == a.veteran_id).row(0, named=True)
        assert (a.name_display, a.modzcta, a.borough) == (
            vet["name_display"], vet["modzcta"], vet["borough"])
        assert a.capacity_bucket == ACTION_COST_UNIT[a.action]
        assert a.message_id == f"msg-{a.action_id}"


def test_actions_never_exceed_capacity_and_ranks_are_dense(client: TestClient) -> None:
    cap = dict(call=7, ride=3, refill=11, booking=2, evac=1, pharmacist_slot=1, va_fill=4)
    resp = api.ActionsResponse.model_validate(_post(client, capacity=cap))
    used: dict[str, int] = {}
    for a in resp.actions:
        used[a.capacity_bucket] = used.get(a.capacity_bucket, 0) + 1
    assert all(n <= resp.capacity[b] for b, n in used.items()), used
    assert used["call"] <= 7 and used.get("evac", 0) <= 1
    assert [a.rank for a in resp.actions] == list(range(1, len(resp.actions) + 1))
    eha = [a.eha for a in resp.actions]
    assert eha == sorted(eha, reverse=True)


def test_a_partial_capacity_keeps_the_defaults_for_the_rest(client: TestClient) -> None:
    resp = api.ActionsResponse.model_validate(_post(client, capacity={"call": 5}))
    assert resp.capacity == {**DEFAULT_CAPACITY, "call": 5}
    assert resp.capacity["ride"] == DEFAULT_CAPACITY["ride"]


def test_more_call_capacity_never_averts_less_harm(client: TestClient) -> None:
    """The slider tells the wrong story if this ever fails."""
    totals = [_post(client, capacity={"call": c})["total_eha"] for c in (0, 5, 10, 20, 40, 80)]
    assert totals == sorted(totals), totals
    assert totals[-1] > totals[0], "the slider has to move the number"


def test_the_same_click_gives_the_same_bytes(client: TestClient) -> None:
    ask = {"date": str(DAY), "capacity": {"call": 13}, "group_floor": {"borough": 0.1}}
    first, second = client.post("/actions", json=ask), client.post("/actions", json=ask)
    assert first.status_code == 200 and first.content == second.content


def test_baselines_are_the_same_team_in_a_different_order(client: TestClient) -> None:
    resp = api.ActionsResponse.model_validate(_post(client))
    by_name = {b.name: b.total_eha for b in resp.baselines}
    assert list(by_name) == ["leeward", "rank_by_age", "rank_by_chronic", "random"]
    assert by_name["leeward"] == resp.total_eha
    for name in ("rank_by_age", "rank_by_chronic", "random"):
        assert 0 < by_name[name] <= by_name["leeward"], (name, by_name)


def test_a_group_floor_is_passed_through(client: TestClient) -> None:
    resp = api.ActionsResponse.model_validate(
        _post(client, capacity={"call": 50}, group_floor={"borough": 0.1}))
    calls = Counter(a.borough for a in resp.actions if a.capacity_bucket == "call")
    boroughs = set(_table("cohort")["borough"].to_list())
    assert sum(calls.values()) <= 50
    assert set(calls) == boroughs and min(calls.values()) >= 5, (
        f"a 10% floor of 50 calls reserves 5 for every borough, got {dict(calls)}")


@pytest.mark.parametrize("body,code,mentions", [
    ({"capacity": {"helicopter": 3}}, 422, "helicopter"),
    ({"capacity": {"call": -1}}, 422, "negative"),
    ({"group_floor": {"shoe_size": 0.1}}, 422, "shoe_size"),
    ({"group_floor": {"borough": 0.9}}, 422, "borough"),
    ({"prior_scale": 0.5}, 422, "prior_scale"),
    ({"scenario": "ida_flash_flood"}, 422, "sandy_then_heat"),
    ({"date": "2031-01-01"}, 404, "2026-07-01"),
])
def test_actions_reject_what_they_cannot_honour(client: TestClient, body, code, mentions) -> None:
    r = client.post("/actions", json={"date": str(DAY), **body})
    assert r.status_code == code and mentions in r.json()["detail"]


def test_actions_reject_a_malformed_body(client: TestClient) -> None:
    assert client.post("/actions", json={}).status_code == 422
    assert client.post("/actions", json={"date": str(DAY), "surprise": 1}).status_code == 422


def _tile(cohort: pl.DataFrame, scores: pl.DataFrame, copies: int):
    """The same people again and again, to reach the panel size the demo runs at."""
    def rename(df: pl.DataFrame, k: int) -> pl.DataFrame:
        return df.with_columns(pl.col("veteran_id") + f"-{k}")
    return (pl.concat([rename(cohort, k) for k in range(copies)]),
            pl.concat([rename(scores, k) for k in range(copies)]))


def test_the_slider_answers_inside_300ms_at_ten_thousand_veterans(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The panel is 10,000 veterans and the slider has to feel instant. The fixtures are, if
    anything, harder than the real scores (more Act-now veterans), so this is not a soft test."""
    _build_data(tmp_path / "data", monkeypatch)
    cohort, scores = _tile(_table("cohort"), _table("scores").filter(pl.col("date") == DAY), 20)
    assert cohort.height == 10_000
    schema.write(cohort, "cohort")
    schema.write(scores, "scores")
    client = TestClient(api_main.app)
    ask = {"date": str(DAY), "capacity": {"call": 40}}
    assert client.post("/actions", json=ask).status_code == 200      # warm the cache

    ms = []
    for _ in range(7):
        t0 = time.perf_counter()
        r = client.post("/actions", json=ask)
        ms.append(1000 * (time.perf_counter() - t0))
        assert r.status_code == 200
    assert api.ActionsResponse.model_validate(r.json()).n_panel == 10_000

    # 300 ms is a product requirement about how the slider feels, and it is measured on the
    # machine the demo runs on. A shared CI runner is not that machine -- this took 127 ms
    # locally and a 329 ms median on GitHub Actions, which says nothing about the code. So
    # CI keeps a budget an order of magnitude looser: still enough to catch an accidental
    # O(n) blowup or a re-fit inside the request, without failing on a noisy neighbour.
    budget = 3000 if os.environ.get("CI") else 300
    where = "CI runner" if os.environ.get("CI") else "this machine"
    assert statistics.median(ms) < budget, (
        f"POST /actions took {sorted(ms)} ms at 10,000 veterans on {where}, "
        f"budget {budget} ms")


# --------------------------------------------------------------------------- #
# GET /message/{action_id}
# --------------------------------------------------------------------------- #

def _fake_messages(seen: list) -> types.ModuleType:
    """Stands in for `leeward.outreach.messages`, which another lane is writing."""
    def render(action, veteran):
        seen.append((action, veteran))
        body = ("VEText. Your phrase is: one two three four. The VA will never ask you to pay, "
                "wire money, or share bank details. VSAFE 833-388-7233. "
                "Veterans Crisis Line: dial 988, press 1.")
        return api.Message(
            message_id=action["message_id"], action_id=action["action_id"],
            veteran_id=veteran["veteran_id"], channel="VEText", addressed_to="veteran",
            verification_phrase="one two three four", body=body, includes_never_pay_line=True,
            includes_vsafe=True, includes_crisis_line=True)
    mod = types.ModuleType("leeward.outreach.messages")
    mod.render = render
    return mod


def test_message_is_503_with_a_reason_until_the_module_lands(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "leeward.outreach.messages", None)   # import raises
    aid = _post(client)["actions"][0]["action_id"]
    r = client.get(f"/message/{aid}")
    assert r.status_code == 503 and "leeward.outreach.messages" in r.json()["detail"]


def test_message_renders_any_action_the_slider_produced(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list = []
    monkeypatch.setitem(sys.modules, "leeward.outreach.messages", _fake_messages(seen))
    cached = _table("actions").filter(pl.col("date") == DAY)["action_id"].to_list()
    # A capacity nobody cached: its action ids exist only because the slider made them.
    slid = [a["action_id"] for a in _post(client, capacity={"call": 3, "ride": 2})["actions"]]
    fresh = next((i for i in slid if i not in set(cached)), None)
    assert fresh is not None, "a smaller capacity should have produced some new action ids"

    for aid in (cached[0], fresh):
        r = client.get(f"/message/{aid}")
        assert r.status_code == 200, r.text
        assert api.Message.model_validate(r.json()).action_id == aid
        action, vet = seen[-1]
        assert action["action_id"] == aid and vet["veteran_id"] == action["veteran_id"]
    assert client.get(f"/message/msg-{fresh}").json()["action_id"] == fresh


def test_message_for_an_unknown_action_is_404(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "leeward.outreach.messages", _fake_messages([]))
    assert client.get("/message/not-an-action").status_code == 404


def test_a_render_that_returns_junk_is_a_500_that_says_so(
        client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    junk = types.ModuleType("leeward.outreach.messages")
    junk.render = lambda action, veteran: {"body": "hello"}
    monkeypatch.setitem(sys.modules, "leeward.outreach.messages", junk)
    aid = _post(client)["actions"][0]["action_id"]
    r = client.get(f"/message/{aid}")
    assert r.status_code == 500 and "did not return a Message" in r.json()["detail"]


def test_the_real_messages_module_carries_all_five_elements_through_the_api(
        client: TestClient) -> None:
    """Skips until `leeward.outreach.messages` lands, then enforces the anti-scam promise
    end to end: what the veteran is sent is what the module rendered, intact."""
    pytest.importorskip("leeward.outreach.messages")
    actions = _post(client)["actions"][:30]
    for a in actions:
        m = api.Message.model_validate(client.get(f"/message/{a['action_id']}").json())
        for element in MANDATORY_MESSAGE_ELEMENTS:
            assert element in m.body, f"{a['action_id']}: message lacks {element!r}"
        assert m.includes_never_pay_line and m.includes_vsafe and m.includes_crisis_line
        assert len(m.verification_phrase.split()) == 4


# --------------------------------------------------------------------------- #
# POST /log
# --------------------------------------------------------------------------- #

def _log_body(action_id: str, vid: str = "SYN-000001", **kw) -> dict:
    return {"action_id": action_id, "veteran_id": vid, "date": str(DAY), "done": True,
            "reached": True, "need_occurred": False, "partner_ack": None,
            "logged_by": "care_team_rahul", **kw}


def test_log_appends_one_row_and_is_append_only(client: TestClient, data: Path) -> None:
    before = _table("outcome_log")
    r = client.post("/log", json=_log_body("new-action-1"))
    assert r.status_code == 200
    assert api.LogResponse.model_validate(r.json()) == api.LogResponse(ok=True,
                                                                        n_rows=before.height + 1)
    after = _table("outcome_log")
    assert after.height == before.height + 1
    row = after.filter(pl.col("action_id") == "new-action-1").row(0, named=True)
    assert (row["veteran_id"], row["done"], row["reached"], row["need_occurred"],
            row["partner_ack"], row["logged_by"], row["date"]) == (
        "SYN-000001", True, True, False, None, "care_team_rahul", DAY)
    assert after.head(before.height).equals(before), "existing rows are untouched"

    again = client.post("/log", json=_log_body("new-action-1", done=False))
    assert again.status_code == 409 and "append-only" in again.json()["detail"]
    assert _table("outcome_log").height == before.height + 1
    assert client.post("/log", json=_log_body("new-action-2")).json()["n_rows"] == \
        before.height + 2


def test_log_rejects_what_it_should_not_record(client: TestClient) -> None:
    h = _table("outcome_log").height
    assert client.post("/log", json=_log_body("a", vid="NOBODY")).status_code == 404
    assert client.post("/log", json=_log_body("b", logged_by="  ")).status_code == 422
    assert client.post("/log", json={"action_id": "c"}).status_code == 422
    assert _table("outcome_log").height == h


# --------------------------------------------------------------------------- #
# GET /report
# --------------------------------------------------------------------------- #

def test_report_before_the_eval_has_run_is_only_the_rung(client: TestClient) -> None:
    rep = api.ReportResponse.model_validate(client.get("/report").json())
    assert rep.model_rung == _table("scores")["model_rung"][0]
    assert rep.generated_at is None, "null is how the UI knows nothing was audited"
    assert rep.fairness == [] and rep.recovery == []


def test_report_shows_a_failed_audit_never_hides_it(client: TestClient, data: Path) -> None:
    failing = api.ReportResponse(
        model_rung=1, rhat_max=1.004, divergences=0, generated_at="2026-09-19T20:00:00",
        fairness=[api.FairnessRow(stratum="borough", group="Bronx", n=90, ece=0.04, fnr=0.42,
                                  fnr_ratio_to_cohort=1.31, flagged=True)],
        fairness_failed=True)
    (data.parent / "report").mkdir(exist_ok=True)
    (data.parent / "report" / "report.json").write_text(failing.model_dump_json())
    try:
        got = api.ReportResponse.model_validate(client.get("/report").json())
        assert got == failing
        assert got.fairness_failed and got.fairness[0].flagged
    finally:
        (data.parent / "report" / "report.json").unlink()


def test_a_report_that_does_not_match_its_model_is_a_500_that_says_so(
        client: TestClient, data: Path) -> None:
    (data.parent / "report").mkdir(exist_ok=True)
    path = data.parent / "report" / "report.json"
    path.write_text('{"model_rung": 9, "calibration": "lots"}')
    try:
        r = client.get("/report")
        assert r.status_code == 500 and "ReportResponse" in r.json()["detail"]
    finally:
        path.unlink()


# --------------------------------------------------------------------------- #
# GET /export
# --------------------------------------------------------------------------- #

def test_export_is_a_csv_of_consenting_veterans_only(client: TestClient) -> None:
    r = client.get("/export", params={"date": str(DAY)})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "partner_sheet_2026-07-16.csv" in r.headers["content-disposition"]
    sheet = pl.read_csv(r.content)
    assert sheet.height > 0, "the fixtures have ride actions for consenting veterans"
    assert sheet.columns == ["date", "veteran_id", "name_display", "modzcta", "borough",
                             "action", "consent_partner_check", "consent_ride",
                             "consent_housing"]
    cohort = _table("cohort")
    no_consent = set(cohort.filter(~pl.col("consent_partner_check"))["veteran_id"])
    assert not set(sheet["veteran_id"]) & no_consent
    assert set(sheet["action"]) <= set(CONSENT_FOR)
    rides = sheet.filter(pl.col("action").is_in(["cooling_center_ride", "clean_air_room",
                                                 "evacuation_assist"]))
    assert rides["consent_ride"].all() and sheet["consent_partner_check"].all()


def test_export_for_an_unscored_day_is_404(client: TestClient) -> None:
    assert client.get("/export", params={"date": "2031-01-01"}).status_code == 404


# --------------------------------------------------------------------------- #
# When the data is not there, or is wrong, the answer says what to do
# --------------------------------------------------------------------------- #

def test_missing_tables_are_503_that_name_the_fix(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(schema, "DATA", tmp_path / "empty")
    client = TestClient(api_main.app)
    for r in (client.get("/scores", params={"date": str(DAY), "need": "heat"}),
              client.post("/actions", json={"date": str(DAY)}),
              client.get("/forecast")):
        assert r.status_code == 503 and "make fixtures" in r.json()["detail"], r.text


def test_a_table_that_breaks_its_contract_is_a_500_that_says_which(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _build_data(tmp_path / "data", monkeypatch)
    bad = pl.DataFrame({"veteran_id": ["SYN-000000"], "date": [DAY], "need": ["heat"]})
    bad.write_parquet(schema.TABLES["scores"].path)     # deliberately not schema.write
    r = TestClient(api_main.app).get("/scores", params={"date": str(DAY), "need": "heat"})
    assert r.status_code == 500 and "[scores]" in r.json()["detail"]


def test_a_table_rewritten_under_a_live_server_is_picked_up(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`make score` re-run while the demo is up must show on the next click, not the next boot."""
    _build_data(tmp_path / "data", monkeypatch)
    client = TestClient(api_main.app)
    first = client.post("/actions", json={"date": str(DAY)}).json()["total_eha"]
    half = _table("scores").with_columns(pl.col("p_mean") / 2, pl.col("p_lo80") / 2,
                                         pl.col("p_hi80") / 2)
    schema.write(half, "scores")
    second = client.post("/actions", json={"date": str(DAY)}).json()["total_eha"]
    assert second < first

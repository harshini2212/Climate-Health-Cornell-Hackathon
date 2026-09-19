"""decision_quality.py -- the Impact bar chart. Written before the module.

Harm averted is `Σ w_k · τ[a,k] · y_true[i,k,t]` over the veterans a strategy selects. The
arithmetic, the three baselines and the outputs are checked here against hand-built frames.
The one test that needs the real allocator skips -- only on the import of
`leeward.decision.allocate` -- until the api lane lands it, and then it has to show Leeward
beating random on a fixture where the answer is not in doubt.
"""

from __future__ import annotations

import importlib
import re
from datetime import date

import polars as pl
import pytest

from leeward import schema
from leeward.api.schemas import DecisionQualityRow
from leeward.eval import decision_quality as dq
from leeward.schema import NEEDS

D1, D2 = date(2026, 7, 2), date(2026, 7, 3)

# Hand-checkable weights: only the numbers the arithmetic tests need.
W = {"breathing": 1.0, "heat": 2.0, "mental": 1.0, "treatment_gap": 5.0, "access_loss": 1.0}
TAU = {"care_team_call": {"treatment_gap": 0.4, "heat": 0.3},
       "early_refill": {"treatment_gap": 0.5}}


def _outcomes(positive: dict[tuple[str, date], set[str]], vets: list[str],
              days: list[date]) -> pl.DataFrame:
    """Every veteran x day x need, y = 1 only where `positive` says so."""
    rows = [{"veteran_id": v, "date": d, "need": k, "y": int(k in positive.get((v, d), set()))}
            for v in vets for d in days for k in NEEDS]
    return pl.DataFrame(rows, schema={"veteran_id": pl.Utf8, "date": pl.Date,
                                      "need": pl.Utf8, "y": pl.Int32})


def _picked(rows: list[tuple[str, date, str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"veteran_id": pl.Utf8, "date": pl.Date, "action": pl.Utf8},
                        orient="row")


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #

def test_harm_averted_is_hand_checkable() -> None:
    out = _outcomes({("A", D1): {"treatment_gap", "heat"},
                     ("A", D2): {"breathing"}},          # a different day: must not count
                    ["A", "B"], [D1, D2])
    picked = _picked([("A", D1, "care_team_call"), ("B", D1, "care_team_call")])
    # A: 5 * 0.4 (treatment gap) + 2 * 0.3 (heat) = 2.6.  B: nothing happened, so 0.
    assert dq.harm_averted(picked, out, W, TAU) == pytest.approx(2.6)


def test_stacked_actions_never_prevent_more_than_the_whole_event() -> None:
    """Two actions on one need combine as 1 - (1-0.4)(1-0.5) = 0.7, not 0.4 + 0.5 = 0.9."""
    out = _outcomes({("C", D1): {"treatment_gap"}}, ["C"], [D1])
    picked = _picked([("C", D1, "care_team_call"), ("C", D1, "early_refill")])
    assert dq.harm_averted(picked, out, W, TAU) == pytest.approx(5 * 0.7)


def test_an_action_with_no_tau_row_averts_nothing() -> None:
    """check_in_call is an information action; SPEC 7.2 gives it no τ."""
    out = _outcomes({("A", D1): set(NEEDS)}, ["A"], [D1])
    assert dq.harm_averted(_picked([("A", D1, "check_in_call")]), out, W, TAU) == 0.0


def test_a_selected_veteran_with_no_outcomes_is_an_error_not_a_zero() -> None:
    """Silently scoring a missing veteran-day as 0 would undercount whichever strategy picked it."""
    out = _outcomes({}, ["A"], [D1])
    with pytest.raises(ValueError, match="no outcomes"):
        dq.harm_averted(_picked([("GHOST", D1, "care_team_call")]), out, W, TAU)


def test_weights_must_cover_every_need() -> None:
    out = _outcomes({}, ["A"], [D1])
    with pytest.raises(ValueError, match="w_k"):
        dq.harm_averted(_picked([("A", D1, "care_team_call")]), out, {"heat": 4.0}, TAU)


# --------------------------------------------------------------------------- #
# The baselines
# --------------------------------------------------------------------------- #

def _panel() -> pl.DataFrame:
    return pl.DataFrame({
        "veteran_id": ["V1", "V2", "V3", "V4", "V5", "V6"],
        "age":        [70,   91,   55,   91,   80,   62],
        "n_chronic":  [2,    1,    7,    0,    3,    7],
    })


def test_rank_by_age_takes_the_oldest_and_breaks_ties_by_id() -> None:
    got = dq.baseline_actions("rank_by_age", _panel(), D1, 3)
    assert got["veteran_id"].to_list() == ["V2", "V4", "V5"]


def test_rank_by_chronic_takes_the_most_conditions() -> None:
    got = dq.baseline_actions("rank_by_chronic", _panel(), D1, 2)
    assert got["veteran_id"].to_list() == ["V3", "V6"]


def test_every_baseline_makes_k_generic_calls_from_todays_panel() -> None:
    panel = _panel()
    for strategy in dq.BASELINES:
        got = dq.baseline_actions(strategy, panel, D1, 4)
        assert got.height == 4 and got["veteran_id"].n_unique() == 4
        assert set(got["veteran_id"]) <= set(panel["veteran_id"])
        assert got["action"].unique().to_list() == [dq.BASELINE_ACTION]
        assert got["date"].unique().to_list() == [D1]
        # more capacity than panel: everyone, once
        assert dq.baseline_actions(strategy, panel, D1, 50).height == panel.height


def test_random_is_seeded_nested_in_k_and_differs_by_day() -> None:
    """Same click, same number on stage -- but not the same six veterans every day."""
    panel = _panel()

    def pick(day: date, k: int) -> list[str]:
        return dq.baseline_actions("random", panel, day, k)["veteran_id"].to_list()

    assert pick(D1, 4) == pick(D1, 4)
    assert pick(D1, 2) == pick(D1, 4)[:2]
    assert pick(D1, 6) != pick(D2, 6)
    # the shuffle must not depend on the order the cohort happens to arrive in
    reversed_panel = panel.reverse()
    assert dq.baseline_actions("random", reversed_panel, D1, 4)["veteran_id"].to_list() \
        == pick(D1, 4)


def test_an_unknown_strategy_is_refused() -> None:
    with pytest.raises(ValueError, match="strategy"):
        dq.baseline_actions("rank_by_vibes", _panel(), D1, 2)


# --------------------------------------------------------------------------- #
# What comes out
# --------------------------------------------------------------------------- #

def _tidy() -> pl.DataFrame:
    rows = []
    for d, bump in ((D1, 0.0), (D2, 2.0)):
        for k in dq.KS:
            for i, s in enumerate(dq.STRATEGIES):
                rows.append({"date": d, "k": k, "strategy": s,
                             "harm_averted": k / 10 * (4 - i) + bump,
                             "n_veterans": k, "n_actions": k})
    return pl.DataFrame(rows, schema=dq.TIDY_SCHEMA)


def test_summary_is_mean_harm_per_day_and_fits_the_report_contract() -> None:
    summary = dq.summarise(_tidy())
    assert summary.height == len(dq.KS) * len(dq.STRATEGIES)
    first = summary.row(0, named=True)
    assert (first["k"], first["strategy"]) == (dq.KS[0], "leeward")
    # leeward at K=20: (8.0 + 10.0) / 2 days
    assert first["harm_averted"] == pytest.approx(9.0)
    for row in summary.to_dicts():
        DecisionQualityRow(k=row["k"], strategy=row["strategy"],
                           harm_averted=row["harm_averted"])


def test_outputs_are_a_tidy_csv_and_an_offline_plotly_chart(tmp_path) -> None:
    csv, html = dq.write_outputs(_tidy(), tmp_path)
    back = pl.read_csv(csv, try_parse_dates=True)
    assert back.columns == list(dq.TIDY_SCHEMA)
    assert back.height == _tidy().height
    page = html.read_text(encoding="utf-8")
    assert "plotly" in page.lower()
    assert not re.search(r"<script[^>]*\bsrc=[\"']https?://", page), (
        "the chart loads plotly.js from the network; it must open with the wifi off")


# --------------------------------------------------------------------------- #
# The one that needs the real allocator
# --------------------------------------------------------------------------- #

def _allocator_or_skip() -> None:
    try:
        importlib.import_module("leeward.decision.allocate")
    except ModuleNotFoundError as e:
        if e.name not in ("leeward.decision", "leeward.decision.allocate"):
            raise   # the allocator exists and one of *its* imports is broken: that is a failure
        pytest.skip("leeward.decision.allocate not built yet -- Leeward-beats-random in "
                    "decision_quality activates the moment it lands")


def _neutral(col: schema.Column):
    if col.values:
        return col.values[0]
    if col.dtype == pl.Boolean:
        return False
    if isinstance(col.dtype, pl.List):
        return []
    lo = col.bounds[0] if col.bounds else 0
    if col.dtype.is_integer():
        return int(lo)
    if col.dtype.is_float():
        return float(lo)
    return ""


def _tiny_world(n: int = 40, at_risk=(4, 11, 23, 31, 38)):
    """40 veterans, 2 days. Five of them are the only ones anything happens to, the model
    sees it coming (p = 0.6, narrow, low epistemic share), and they are the youngest with
    the fewest conditions -- so age and chronic-count ranking cannot find them by accident.
    `at_risk` is chosen so random (seed 0) misses them on both days at K = 3 and 5.
    """
    ids = [f"T{i:02d}" for i in range(n)]
    risky = {ids[i] for i in at_risk}
    base = {c.name: _neutral(c) for c in schema.TABLES["cohort"].columns}
    base |= {"modzcta": "10463", "borough": "Bronx", "facility_id": "526", "floor": "upper",
             "caregiver": "informal_coresident", "income_band": "mid", "hvi": 1,
             "home_ac": True, "name_display": "Tiny Fixture"}
    flags = {f"{c.name}_synthetic": True for c in schema.TABLES["cohort"].columns if c.synthetic}
    cohort = pl.DataFrame([
        base | flags | {"veteran_id": v,
                        "age": 30 if v in risky else 66 + i % 25,
                        "n_chronic": 0 if v in risky else 2 + i % 6}
        for i, v in enumerate(ids)
    ], schema_overrides={c.name: c.dtype for c in schema.TABLES["cohort"].columns})

    days = [D1, D2]
    scores = pl.DataFrame([
        {"veteran_id": v, "date": d, "need": k,
         "p_mean": 0.6 if v in risky else 0.005,
         "p_lo80": 0.5 if v in risky else 0.002,
         "p_hi80": 0.7 if v in risky else 0.010,
         "p_epistemic_share": 0.05,
         "driver_1": "tiny fixture", "driver_2": None, "driver_3": None,
         "driver_1_contrib": 1.0, "driver_2_contrib": None, "driver_3_contrib": None,
         "model_rung": 0}
        for v in ids for d in days for k in NEEDS
    ], schema_overrides={c.name: c.dtype for c in schema.TABLES["scores"].columns})
    outcomes = _outcomes({(v, d): set(NEEDS) for v in risky for d in days}, ids, days)

    for df, name in ((cohort, "cohort"), (scores, "scores"), (outcomes, "outcomes")):
        schema.validate(df, name)
    return scores, cohort, outcomes


def test_the_tiny_fixture_is_one_random_cannot_luck_into() -> None:
    """Guards the next test: if random found the at-risk five, 'beats random' would be noise."""
    scores, cohort, outcomes = _tiny_world()
    for d in (D1, D2):
        for k in (3, 5):
            for strategy in dq.BASELINES:
                picked = dq.baseline_actions(strategy, cohort, d, k)
                assert dq.harm_averted(picked, outcomes, W, TAU) == 0.0, (strategy, d, k)


def test_leeward_beats_every_baseline_on_the_tiny_fixture() -> None:
    scores, cohort, outcomes = _tiny_world()
    _allocator_or_skip()
    w, tau = dq.decision_weights()          # the same w_k and τ the allocator optimises
    tidy = dq.evaluate(scores, cohort, outcomes, w=w, tau=tau, ks=(3, 5))

    assert tidy.height == 2 * 2 * len(dq.STRATEGIES)
    wide = tidy.pivot(on="strategy", index=["date", "k"], values="harm_averted")
    for row in wide.to_dicts():
        assert row["leeward"] > 0, f"Leeward averted nothing on {row['date']} at K={row['k']}"
        for b in dq.BASELINES:
            assert row["leeward"] > row[b], f"{b} beat Leeward on {row['date']} at K={row['k']}"
    lee = tidy.filter(pl.col("strategy") == "leeward")
    assert (lee["n_actions"] <= lee["k"]).all(), "Leeward used more calls than it was given"

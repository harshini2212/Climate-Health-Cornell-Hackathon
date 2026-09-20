"""Guardrails: the project's non-negotiables, enforced by machine.

Nobody is reading the diffs, so these tests are the review. They are deliberately
*semantic* -- they check that the numbers mean the right thing, not merely that the code
ran. An agent producing plausible-but-wrong work fails here.

Each guardrail for a module that does not exist yet **skips with a message naming what it
will enforce**, and starts enforcing automatically the moment that module lands. So the
suite gets stricter as the build fills in, with nobody having to remember to turn it on.

Run `make check` and read the skip list: it is an accurate to-do list.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

import polars as pl
import pytest

from leeward import schema
from leeward.schema import DEFAULT_CAPACITY, MANDATORY_MESSAGE_ELEMENTS, NEEDS
from tables import table

ROOT = Path(__file__).resolve().parents[1]


def _mod(name: str):
    """Import a lane's module, or skip with a message saying what this will check."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip(f"{name} not built yet -- this guardrail activates the moment it lands")


# --------------------------------------------------------------------------- #
# Contracts. These never skip: the suite builds its own frames (tests/tables.py).
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(schema.TABLES))
def test_every_table_matches_its_contract(name: str) -> None:
    schema.validate(table(name), name)


def test_no_test_reads_a_contract_table_out_of_data() -> None:
    """`data/` is pipeline state, not test input, and the gate must not depend on it.

    `make fixtures` leaves 500 fake veterans there; `make cohort` replaces some of it with
    10,000 real ones; in between it is a mix. A suite that reads it goes red depending on
    which target ran last, and the next person spends the night hunting a product bug that
    was never there. Frames come from `tests/tables.py`; `data/reference/` stays fair game,
    because it is committed input rather than anything a make target regenerates.
    """
    # A contract table is `data/<name>.parquet` at the top level, or TABLES[...].path.
    # `data/reference/` and `data/raw/` are committed or upstream input -- no make target
    # regenerates them into a contract table -- so reading those is stable and allowed.
    banned = re.compile(r"""schema\.TABLES\[[^\]]+\]\.path"""
                        r"""|schema\.DATA\s*/\s*["'][^"'/]+\.parquet["']""")
    # A file that monkeypatches `schema.DATA` has redirected the data root to a tmp_path,
    # so those writes never touch the real directory -- that is the isolation this guardrail
    # is asking for, not a violation of it. Exempting per file rather than per line is
    # coarse, but the alternative is parsing scope, and the failure it would miss (a file
    # that isolates in one test and reads the real data/ in another) is caught by the rest
    # of the suite going red depending on which make target ran last.
    isolates = re.compile(r"monkeypatch\.setattr\(\s*schema\s*,\s*[\"']DATA[\"']"
                          r"|mp\.setattr\(\s*schema\s*,\s*[\"']DATA[\"']")
    offenders = []
    for path in sorted((ROOT / "tests").glob("*.py")):
        src = path.read_text(encoding="utf-8")
        if banned.search(src) and not isolates.search(src):
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} read a contract table out of data/; use `from tables import table`")


def test_synthetic_columns_all_carry_their_flag() -> None:
    """'Only the people are synthetic' is the central claim. It has to be enforced."""
    cohort = table("cohort")
    for col in schema.TABLES["cohort"].columns:
        if not col.synthetic:
            continue
        flag = f"{col.name}_synthetic"
        assert flag in cohort.columns, f"{col.name} is synthetic but has no {flag} column"
        assert cohort[flag].all(), f"{flag} is False somewhere; synthetic data must say so"


def test_intervals_are_ordered() -> None:
    s = table("scores")
    bad = s.filter((pl.col("p_lo80") > pl.col("p_mean")) | (pl.col("p_mean") > pl.col("p_hi80")))
    assert bad.height == 0, f"{bad.height} scores where lo80 <= mean <= hi80 is violated"


def test_every_veteran_day_has_all_five_needs() -> None:
    s = table("scores")
    per = s.group_by("veteran_id", "date").agg(pl.col("need").n_unique().alias("k"))
    assert per["k"].min() == len(NEEDS), (
        "some veteran-days are missing needs; the care-team list would silently under-rank them")


def test_geography_key_is_real_nyc() -> None:
    """A cohort in ZIPs that do not exist renders an empty map and nobody notices until demo."""
    real = set(pl.read_parquet(schema.REFERENCE / "nyc_modzcta.parquet")["modzcta"].to_list())
    for name in ("cohort", "hazards"):
        got = set(table(name)["modzcta"].unique().to_list())
        assert got <= real, f"{name} references ZIPs not in nyc_modzcta: {sorted(got - real)[:5]}"


def test_facilities_are_real_stations() -> None:
    real = set(pl.read_parquet(
        schema.REFERENCE / "va_facilities_nyc_hazard.parquet")["station_no"].to_list())
    for name in ("cohort", "site_status"):
        col = "facility_id"
        got = set(table(name)[col].unique().to_list())
        assert got <= real, f"{name}.{col} has unknown stations: {sorted(got - real)[:5]}"


# --------------------------------------------------------------------------- #
# Decision layer
# --------------------------------------------------------------------------- #

def test_capacity_is_never_exceeded() -> None:
    """The whole pitch is 'cut at the team's real capacity'. Overfilling it is a lie."""
    actions = table("actions")
    for day, grp in actions.group_by("date"):
        used = grp.group_by("capacity_bucket").agg(pl.len().alias("n"))
        for bucket, n in zip(used["capacity_bucket"], used["n"], strict=False):
            cap = DEFAULT_CAPACITY.get(bucket)
            assert cap is not None, f"unknown capacity bucket {bucket!r}"
            assert n <= cap, f"{day}: {n} actions in bucket {bucket}, capacity is {cap}"


def test_one_action_per_veteran_unless_act_now() -> None:
    actions = table("actions")
    counts = (actions.group_by("date", "veteran_id")
                     .agg(pl.len().alias("n"), pl.col("tier").min().alias("tier")))
    over = counts.filter((pl.col("n") > 1) & (pl.col("tier") != "act_now"))
    assert over.height == 0, f"{over.height} non-act-now veterans got more than one action"
    assert counts["n"].max() <= 3, "no veteran may receive more than three actions in a day"


def test_ranks_are_dense_and_ordered_by_eha() -> None:
    actions = table("actions")
    for _, grp in actions.group_by("date"):
        g = grp.sort("rank")
        assert g["rank"].to_list() == list(range(1, g.height + 1)), "ranks must be 1..n, no gaps"
        eha = g["eha"].to_list()
        assert eha == sorted(eha, reverse=True), "rank must be non-increasing in EHA"


def test_more_capacity_never_averts_less_harm() -> None:
    """A monotonicity property. If this breaks, the capacity slider tells the wrong story."""
    allocate = _mod("leeward.decision.allocate")
    fn = getattr(allocate, "allocate", None)
    assert fn is not None, "leeward.decision.allocate must expose allocate(...)"
    sig = inspect.signature(fn)
    assert "capacity" in sig.parameters, "allocate() must take a `capacity` dict"

    total = getattr(allocate, "total_eha", None)
    if total is None:
        pytest.skip("allocate.total_eha(actions) not exposed yet")
    scores, cohort = table("scores"), table("cohort")
    prev = None
    for calls in (10, 20, 40, 80):
        cap = dict(DEFAULT_CAPACITY, call=calls)
        got = total(fn(scores=scores, cohort=cohort, capacity=cap))
        if prev is not None:
            assert got >= prev - 1e-9, f"capacity {calls} averted less harm than the step below"
        prev = got


# --------------------------------------------------------------------------- #
# Outreach. These are the promises made to a veteran, so they are hard failures.
# --------------------------------------------------------------------------- #

def test_every_message_carries_all_mandatory_elements() -> None:
    messages = _mod("leeward.outreach.messages")
    render = getattr(messages, "render", None)
    assert render is not None, "leeward.outreach.messages must expose render(...)"

    actions, cohort = table("actions"), table("cohort")
    checked = 0
    for row in actions.head(40).to_dicts():
        vet = cohort.filter(pl.col("veteran_id") == row["veteran_id"])
        if vet.height == 0:
            continue
        msg = render(action=row, veteran=vet.to_dicts()[0])
        body = msg if isinstance(msg, str) else getattr(msg, "body", str(msg))
        for element in MANDATORY_MESSAGE_ELEMENTS:
            assert element in body, (
                f"message for action {row['action_id']} is missing {element!r}. "
                "Every message must be distinguishable from a scam.")
        phrase = getattr(msg, "verification_phrase", None)
        if phrase is not None:
            assert len(phrase.split()) == 4, "the verification phrase is four words"
        checked += 1
    assert checked > 0, "no messages were rendered, so nothing was actually checked"


def test_no_message_contains_an_unapproved_phone_or_shortener() -> None:
    messages = _mod("leeward.outreach.messages")
    allowed = {"833-388-7233", "988", "911"}
    src = Path(inspect.getfile(messages)).read_text(encoding="utf-8")
    for phone in re.findall(r"\b\d{3}-\d{3}-\d{4}\b", src):
        assert phone in allowed, f"unapproved phone number in messages.py: {phone}"
    for bad in ("bit.ly", "tinyurl", "t.co/", "goo.gl"):
        assert bad not in src, f"URL shortener {bad!r} in outreach; scammers use those"


def test_partner_export_excludes_rather_than_redacts() -> None:
    export = _mod("leeward.outreach.export")
    fn = getattr(export, "partner_sheet", None)
    if fn is None:
        pytest.skip("export.partner_sheet(...) not exposed yet")
    cohort, actions = table("cohort"), table("actions")
    sheet = fn(actions=actions, cohort=cohort)
    ids = set(sheet["veteran_id"].to_list()) if hasattr(sheet, "to_list") or hasattr(
        sheet, "columns") else set()
    withheld = set(cohort.filter(~pl.col("consent_partner_check"))["veteran_id"].to_list())
    assert not (ids & withheld), (
        "veterans without partner consent appear in the export; rows are excluded, never redacted")


# --------------------------------------------------------------------------- #
# Honesty guardrails
# --------------------------------------------------------------------------- #

def test_fairness_audit_is_never_suppressed() -> None:
    """A failing audit is displayed. Code that swallows it is the one thing we will not ship."""
    fairness = _mod("leeward.eval.fairness")
    src = Path(inspect.getfile(fairness)).read_text(encoding="utf-8")
    for pattern in (r"except\s*:\s*\n\s*pass", r"except Exception:\s*\n\s*pass"):
        assert not re.search(pattern, src), "fairness.py swallows an exception"
    assert "flagged" in src, "fairness.py must mark groups whose relative FNR gap exceeds 20 pct"


def test_the_fairness_audit_reports_what_a_flag_count_cannot() -> None:
    """The other way to suppress an audit is to report a number that cannot move.

    At a pooled FNR near 1 no group can exceed the 20 percent bar -- clearing it would take
    an FNR above 1 -- so a report that says only "0 of 33 flagged" has shown a metric with
    no failing state and called it a pass. The audit therefore has to keep the raw rate, and
    also carry the direction and the coverage that *can* separate groups.
    """
    fairness = _mod("leeward.eval.fairness")
    if fairness is None:
        pytest.skip("leeward.eval.fairness not built yet; will enforce reach/coverage/"
                    "direction beside the flag, and that raw fnr stays in the report")

    assert "fnr" in fairness.AUDIT_SCHEMA, "the raw rate is the honest denominator"
    assert "fnr" in fairness.REPORT_COLUMNS, "dropping fnr for the nicer number is suppression"
    for column in ("reach_ratio_to_cohort", "coverage", "direction"):
        assert column in fairness.REPORT_COLUMNS, (
            f"{column} must reach report.json; the flag alone cannot report this cohort")

    tbl = fairness.audit(table("scores"), table("outcomes"), table("cohort"))
    pooled = fairness.cohort_fnr(tbl)
    note = fairness.ceiling_note(tbl)
    assert f"{pooled:.3f}" in note, "the pooled rate the ratios divide by is never hidden"
    if not fairness.flag_is_reachable(pooled):
        assert not tbl["flagged"].any(), "a bar above an FNR of 1 cannot be cleared"
        assert "impossible" in note, (
            "a report where nothing can flag has to say so next to the flag count")


def test_model_rung_is_recorded() -> None:
    """You have to be able to say on stage which rung actually fitted."""
    s = table("scores")
    assert s["model_rung"].n_unique() == 1, "scores mix model rungs; that cannot be explained"
    assert s["model_rung"][0] in (0, 1, 2, 3)


def test_demo_path_makes_no_network_calls() -> None:
    """`make demo` must work with the wifi off. Catch an import-time fetch before the judges do."""
    offenders = []
    for path in (ROOT / "leeward").rglob("*.py"):
        if "ingest" in path.parts:      # ingest is allowed to fetch; the demo never calls it
            continue
        src = path.read_text(encoding="utf-8")
        for pattern in (r"\brequests\.(get|post)\b", r"\burllib\.request\b", r"\bhttpx\.(get|post)\b"):
            if re.search(pattern, src):
                offenders.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert not offenders, (
        "network calls outside leeward/ingest/: " + "; ".join(offenders) +
        ". The demo runs offline; fetching belongs in scripts/fetch_sources.py.")


def test_no_real_patient_data_paths() -> None:
    """The one hard rule in CLAUDE.md. Cheap to check, catastrophic to miss."""
    banned = re.compile(r"\b(vistA_prod|phi_|real_patient|mpi_|ssn|social_security)\b", re.I)
    for path in (ROOT / "leeward").rglob("*.py"):
        hit = banned.search(path.read_text(encoding="utf-8"))
        assert not hit, f"{path.relative_to(ROOT)} references {hit.group(0)!r}"


def test_seeds_are_fixed_everywhere_that_randomises() -> None:
    """The same click must produce the same number in rehearsal and on stage."""
    offenders = []
    for path in list((ROOT / "leeward").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        src = path.read_text(encoding="utf-8")
        if re.search(r"np\.random\.default_rng\(\s*\)", src):
            offenders.append(f"{path.relative_to(ROOT)}: default_rng() with no seed")
        if re.search(r"\brandom\.(random|choice|randint|shuffle)\(", src) and "seed" not in src:
            offenders.append(f"{path.relative_to(ROOT)}: stdlib random with no seed")
    assert not offenders, "unseeded randomness: " + "; ".join(offenders)


# --------------------------------------------------------------------------- #
# Cross-lane interfaces.
#
# Lanes are built in parallel by agents that never see each other's code, so the
# failure mode is not a bad function -- it is two correct functions that disagree
# about a name. That surfaces at merge, which is the worst time. These tests pin
# the public surface each lane promises the others, so drift fails in the lane
# that caused it rather than in whoever merges last.
# --------------------------------------------------------------------------- #

DECISION_INTERFACE = {
    "leeward.decision.severity": [("load", dict), ("vector", None)],
    "leeward.decision.tau": [("load", dict), ("matrix", None)],
    "leeward.decision.eha": [("eha_matrix", None), ("voi", None), ("needs_wide", None)],
    "leeward.decision.allocate": [("allocate", None), ("total_eha", None)],
    "leeward.decision.tiers": [("assign", None)],
}


@pytest.mark.parametrize("module,expected", sorted(DECISION_INTERFACE.items()))
def test_decision_layer_public_interface(module: str, expected: list) -> None:
    mod = _mod(module)
    for name, returns in expected:
        fn = getattr(mod, name, None)
        assert callable(fn), (
            f"{module}.{name}() is missing. Other lanes import it; renaming it breaks them "
            f"at merge time. If the interface must change, change it here first.")
        if returns is dict:
            got = fn()
            assert isinstance(got, dict) and got, f"{module}.{name}() must return a non-empty dict"


def test_severity_and_tau_are_yaml_backed_not_hardcoded() -> None:
    """SPEC 7.1 and 7.2: a clinician retunes these without touching code."""
    for mod_name, fname in (("leeward.decision.severity", "severity.yaml"),
                            ("leeward.decision.tau", "tau.yaml")):
        mod = _mod(mod_name)
        path = getattr(mod, "PATH", None)
        assert path is not None and Path(path).name == fname, f"{mod_name}.PATH must point at {fname}"
        assert Path(path).exists(), f"{path} is missing"


def test_severity_covers_every_need_and_tau_every_action() -> None:
    severity = _mod("leeward.decision.severity")
    tau = _mod("leeward.decision.tau")
    w = severity.load()
    assert set(w) >= set(NEEDS), f"severity weights missing needs {set(NEEDS) - set(w)}"
    assert all(v > 0 for v in w.values()), "a severity weight of zero silently drops a need"

    t = tau.load()
    unknown = set(t) - set(schema.ACTION_COST_UNIT)
    assert not unknown, f"tau.yaml names actions the schema does not know: {sorted(unknown)}"
    for action, row in t.items():
        bad = {k: v for k, v in row.items() if not 0.0 <= v <= 1.0}
        assert not bad, f"tau[{action}] has values outside [0,1]: {bad}"

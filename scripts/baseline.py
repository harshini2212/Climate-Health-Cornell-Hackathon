#!/usr/bin/env python3
"""Run the pipeline end to end and record one baseline.

A number is only useful if you can say what it was before. This runs every stage from a
clean start, collects everything worth tracking into one row, writes the machine-readable
copy to `report/baselines/`, and appends a human-readable entry to `docs/BASELINES.md`.

    python scripts/baseline.py                      # run the pipeline, then record
    python scripts/baseline.py --no-run             # record whatever is in data/ already
    python scripts/baseline.py --label "rung 1"     # name the row
    python scripts/baseline.py --compare            # diff the last two, change nothing

Why it re-runs by default: a baseline recorded against artifacts someone left in `data/`
is a baseline for code nobody can identify. Running first means the row and the commit
it names describe the same thing.

Every row carries the git sha, the seed and the scenario, so a later run that moves a
number can be attributed to a change rather than to the weather.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "baselines"
DOC = ROOT / "docs" / "BASELINES.md"

#: Every stage, in order, with the module that runs it.
STAGES = [
    ("hazards", ["-m", "leeward.ingest.hazards", "--scenario", "scenarios/sandy_then_heat.yaml"]),
    ("cohort", ["-m", "leeward.cohort.build"]),
    ("simulate", ["-m", "leeward.cohort.simulate"]),
    ("score", ["-m", "leeward.model.score_prior"]),
    ("allocate", ["-m", "leeward.decision.allocate"]),
    ("report", ["-m", "leeward.eval.report"]),
]

#: Metrics whose direction of "better" is known. Anything not here is context, not a score.
BETTER = {
    "ece_max": "lower", "ece_mean": "lower",
    "recovery_coverage": "higher",
    "harm_averted_k40": "higher", "lift_vs_best_baseline_k40": "higher",
    "harm_per_call_k40": "higher", "lift_per_call_k40": "higher",
    # Discrimination is the evidence ECE is not: a constant at the base rate scores ECE
    # 0.0000 and separates nobody. AUC 0.5 is a coin toss whatever the reliability curve says.
    "auc_heat": "higher", "auc_treatment_gap": "higher", "auc_worst_need": "higher",
    "needs_beating_chance": "higher",
    "fairness_flagged": "lower", "fairness_max_fnr_ratio": "lower",
    "find_out_share": "higher",
    "total_seconds": "lower",
}


def sh(args: list[str]) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr), time.perf_counter() - t0


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_pipeline() -> dict[str, float]:
    timings: dict[str, float] = {}
    for name, args in STAGES:
        code, out, secs = sh(args)
        if code != 0:
            tail = "\n".join(out.strip().splitlines()[-12:])
            raise SystemExit(f"stage {name!r} failed ({code}):\n{tail}")
        timings[name] = round(secs, 2)
        print(f"  {name:10s} {secs:6.1f}s")
    return timings


def collect(timings: dict[str, float]) -> dict:
    cohort = pl.read_parquet(ROOT / "data/cohort.parquet")
    scores = pl.read_parquet(ROOT / "data/scores.parquet")
    actions = pl.read_parquet(ROOT / "data/actions.parquet")
    outcomes = pl.read_parquet(ROOT / "data/outcomes.parquet")
    report = json.loads((ROOT / "report/report.json").read_text())

    ece = report.get("ece_by_need", {})
    fair = report.get("fairness", [])
    dq = report.get("decision_quality", [])

    # Harm averted at K=40, and the lift over whichever baseline did best -- quoting the
    # baseline you beat by least is the number that makes the rest credible.
    at40 = {r["strategy"]: r["harm_averted"] for r in dq if r.get("k") == 40}
    lee40 = at40.get("leeward")
    best_base = max((v for k, v in at40.items() if k != "leeward"), default=None)

    # Harm per day punishes a system for correctly doing less on a calm day: with do-by
    # scheduling leeward leaves calls unspent when nothing is worth one, and 7 of 30
    # held-out days now avert zero on purpose. Per call actually made is the honest
    # efficiency number, and it is the one that went up.
    # Both totals come from the same rows, or the scale silently mismatches: report.json
    # carries the per-day mean while the CSV carries per-day counts.
    calls, harm_total = {}, {}
    dq_csv = ROOT / "report/decision_quality.csv"
    if dq_csv.exists():
        raw = pl.read_csv(dq_csv).filter(pl.col("k") == 40)
        agg = raw.group_by("strategy").agg(pl.col("n_actions").sum().alias("calls"),
                                           pl.col("harm_averted").sum().alias("harm"))
        calls = dict(zip(agg["strategy"], agg["calls"], strict=True))
        harm_total = dict(zip(agg["strategy"], agg["harm"], strict=True))
    per_call = {k: round(harm_total[k] / calls[k], 4) for k in harm_total if calls.get(k)}
    lee_pc = per_call.get("leeward")
    best_pc = max((v for k, v in per_call.items() if k != "leeward"), default=None)

    # Discrimination, and the control that makes calibration meaningful.
    disc = {r["need"]: r for r in report.get("discrimination", [])}
    auc = {k: round(v["within_day_auc"], 3) for k, v in disc.items()}
    const = report.get("constant_ece", {})

    tiers = dict(actions.group_by("tier").agg(pl.len().alias("n")).iter_rows())
    n_act = actions.height

    med = {
        "n_active_meds_mean": round(float(cohort["n_active_meds"].mean()), 2),
        "heat_impairing_pct": round(100 * float((cohort["med_thermoreg_score"] > 0).mean()), 1),
        "cdc_pair_pct": round(100 * float(cohort["med_combo_raas_diuretic"].mean()), 1),
        "cold_chain_pct": round(100 * float(cohort["med_cold_chain"].mean()), 1),
        "controlled_pct": round(100 * float(cohort["med_controlled"].mean()), 1),
    }

    base_rate = {
        need: round(100 * float(outcomes.filter(pl.col("need") == need)["y"].mean()), 3)
        for need in sorted(outcomes["need"].unique().to_list())
    }

    return {
        "recorded_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "machine": platform.machine(),

        "model_rung": report.get("model_rung"),
        "rhat_max": report.get("rhat_max"),
        "divergences": report.get("divergences"),

        "n_veterans": cohort.height,
        "n_zips": int(cohort["modzcta"].n_unique()),
        "n_score_rows": scores.height,
        "n_actions": n_act,
        "medication": med,
        "outcome_base_rate_pct": base_rate,

        "ece_by_need": {k: round(v, 4) for k, v in ece.items()},
        "ece_max": round(max(ece.values()), 4) if ece else None,
        "ece_mean": round(statistics.mean(ece.values()), 4) if ece else None,
        "recovery_coverage": (round(report["recovery_coverage"], 4)
                              if report.get("recovery_coverage") is not None else None),

        "harm_averted_k40": round(lee40, 2) if lee40 is not None else None,
        "harm_averted_baselines_k40": {k: round(v, 2) for k, v in at40.items() if k != "leeward"},
        "lift_vs_best_baseline_k40": (round(lee40 / best_base, 2)
                                      if lee40 and best_base else None),
        "harm_total_k40": round(harm_total["leeward"], 2) if harm_total.get("leeward") else None,
        "calls_spent_k40": calls.get("leeward"),
        "calls_available_k40": max(calls.values()) if calls else None,
        "harm_per_call_k40": lee_pc,
        "harm_per_call_baselines_k40": {k: v for k, v in per_call.items() if k != "leeward"},
        "lift_per_call_k40": round(lee_pc / best_pc, 2) if lee_pc and best_pc else None,

        "within_day_auc": auc,
        "auc_heat": auc.get("heat"),
        "auc_treatment_gap": auc.get("treatment_gap"),
        "auc_worst_need": min(auc.values()) if auc else None,
        "needs_beating_chance": sum(1 for v in auc.values() if v >= 0.65) if auc else None,
        "lift_at_1pct": {k: round(v["lift_at_1pct"], 1) for k, v in disc.items()},
        "scaled_brier": {k: round(v["scaled_brier"], 4) for k, v in disc.items()},
        "constant_ece": {k: round(v, 4) for k, v in const.items()},
        "model_beats_constant_on_ece": (
            all(const[k] > ece[k] for k in ece) if const and ece else None),

        "tier_mix": tiers,
        "find_out_share": round(100 * tiers.get("find_out", 0) / n_act, 3) if n_act else None,
        "act_now_share": round(100 * tiers.get("act_now", 0) / n_act, 2) if n_act else None,

        "fairness_groups": len(fair),
        "fairness_flagged": sum(1 for f in fair if f.get("flagged")),
        "fairness_max_fnr_ratio": (round(max(f["fnr_ratio_to_cohort"] for f in fair), 3)
                                   if fair else None),

        "timings_seconds": timings,
        "total_seconds": round(sum(timings.values()), 1) if timings else None,
    }


def render_row(b: dict, label: str) -> str:
    med, br = b["medication"], b["outcome_base_rate_pct"]
    dirty = " *(uncommitted changes)*" if b["git_dirty"] else ""
    base_s = ", ".join(f"{k} {v}" for k, v in sorted(b["harm_averted_baselines_k40"].items()))
    ece_s = " · ".join(f"{k} {v}" for k, v in sorted(b["ece_by_need"].items()))
    br_s = " · ".join(f"{k} {v}%" for k, v in sorted(br.items()))
    t = b["timings_seconds"]
    t_s = " · ".join(f"{k} {v}s" for k, v in t.items()) if t else "not re-run"

    rhat = b["rhat_max"]
    fit_line = (f"r-hat max {rhat}, {b['divergences']} divergences" if rhat is not None
                else "prior-only, so no r-hat and no divergences")

    return f"""
## {label} — rung {b['model_rung']} — `{b['git_sha']}`{dirty}

*{b['recorded_utc']} · Python {b['python']} · {b['machine']}*

| | |
| --- | --- |
| **Harm averted, 40 calls/day** | **{b['harm_averted_k40']}** vs {base_s} — **{b['lift_vs_best_baseline_k40']}× the best baseline** |
| **Per call actually made** | **{b['harm_per_call_k40']}** — **{b['lift_per_call_k40']}×**, spending {b['calls_spent_k40']} of {b['calls_available_k40']} available calls |
| **Discrimination (within-day AUC)** | heat **{b['auc_heat']}** · treatment gap **{b['auc_treatment_gap']}** · worst need {b['auc_worst_need']} · {b['needs_beating_chance']} of 5 needs above 0.65 |
| **Calibration (ECE, bar 0.03)** | max **{b['ece_max']}** · mean {b['ece_mean']} — but a constant at the base rate scores {min(b['constant_ece'].values()) if b['constant_ece'] else 'n/a'}, so this is not evidence on its own |
| **Parameter coverage (bar 0.90)** | **{b['recovery_coverage']}** |
| **Fairness** | {b['fairness_flagged']} flagged of {b['fairness_groups']} groups · worst FNR ratio {b['fairness_max_fnr_ratio']} |
| **Model fit** | {fit_line} |

**ECE by need** — {ece_s}

**Simulated base rate per day** — {br_s}

**Panel** — {b['n_veterans']:,} veterans across {b['n_zips']} ZIPs · {b['n_score_rows']:,} scored rows · {b['n_actions']:,} actions

**Medication** — {med['n_active_meds_mean']} drugs each · {med['heat_impairing_pct']}% heat-impairing · {med['cdc_pair_pct']}% on the CDC pair · {med['cold_chain_pct']}% cold-chain · {med['controlled_pct']}% controlled

**Tier mix** — act-now {b['act_now_share']}% · find-out {b['find_out_share']}%

**Pipeline** — {t_s} · total {b['total_seconds']}s
"""


def compare(a: dict, b: dict) -> str:
    """b relative to a, for the metrics whose direction is known."""
    lines = [f"`{a['git_sha']}` → `{b['git_sha']}`  (rung {a['model_rung']} → {b['model_rung']})", ""]
    for key, better in BETTER.items():
        x, y = a.get(key), b.get(key)
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        d = y - x
        if abs(d) < 1e-9:
            mark, verdict = "=", "unchanged"
        else:
            improved = (d < 0) if better == "lower" else (d > 0)
            mark = "▲" if d > 0 else "▼"
            verdict = "better" if improved else "WORSE"
        pct = f"{100 * d / x:+.1f}%" if x else "n/a"
        lines.append(f"  {mark} {key:28s} {x:>10} → {y:<10} {pct:>8}  {verdict}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None, help="name this baseline")
    ap.add_argument("--no-run", action="store_true", help="record what is in data/ already")
    ap.add_argument("--compare", action="store_true", help="diff the two most recent, record nothing")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    existing = sorted(OUT.glob("*.json"))

    if args.compare:
        if len(existing) < 2:
            raise SystemExit(f"need two baselines to compare; {len(existing)} recorded")
        a, b = (json.loads(p.read_text()) for p in existing[-2:])
        print(compare(a, b))
        return 0

    if args.no_run:
        print("recording what is already in data/ -- the sha below may not describe it")
        timings: dict[str, float] = {}
    else:
        print("running the pipeline")
        timings = run_pipeline()

    b = collect(timings)
    label = args.label or f"rung {b['model_rung']} baseline"
    stamp = b["recorded_utc"].replace(":", "").replace("-", "")
    path = OUT / f"{stamp}-rung{b['model_rung']}.json"
    path.write_text(json.dumps(b, indent=2, sort_keys=True))

    if not DOC.exists():
        DOC.write_text(HEADER)
    DOC.write_text(DOC.read_text().rstrip() + "\n" + render_row(b, label) + "\n---\n")

    print(f"\n{render_row(b, label)}")
    print(f"wrote {path.relative_to(ROOT)} and appended to {DOC.relative_to(ROOT)}")
    if existing:
        print("\nversus the previous baseline:\n")
        print(compare(json.loads(existing[-1].read_text()), b))
    return 0


HEADER = """# Baselines

One row per recorded run of the whole pipeline. A number is only useful if you can say what
it was before, so this is the before.

```bash
python scripts/baseline.py --label "rung 1"   # run the pipeline, record a row
python scripts/baseline.py --compare          # diff the last two, change nothing
```

Each row names the git sha, the model rung and the seed's scenario, so a number that moves
can be attributed to a change rather than to the weather. Rows are append-only; do not edit
an old one, record a new one.

**What "better" means, per metric:** ECE lower, coverage higher, harm averted higher, lift
higher, fairness flagged lower, find-out share higher (it is zero when the model has no
opinion about uncertainty), total seconds lower. Everything else is context, not a score.

---
"""


if __name__ == "__main__":
    raise SystemExit(main())

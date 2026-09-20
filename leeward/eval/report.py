"""The eval harness, assembled into `report/report.json`. SPEC §11.

`GET /report` serves this file as-is and the Model report screen draws it, so everything
here is shaped by `ReportResponse` in `leeward/api/schemas.py` -- a key that contract has no
room for would 500 the route rather than appear on the screen.

What it says, and what it refuses to say:

- `model_rung` is read off the scores, not asserted here. Scores that mix rungs are an
  error: "which model produced this" has to have one answer on stage.
- `rhat_max` and `divergences` are null at a rung with no MCMC. "We did not sample" and
  "it sampled perfectly" must not look the same.
- Recovery at rung 0 is prior coverage rather than parameter recovery. The contract has
  nowhere to put that word, so it is carried by `model_rung: 0` with a null `rhat_max`, and
  said plainly in `report/recovery.csv`, in the chart's subtitle and on the console.
- **`constant_ece` ships beside `ece_by_need`, and it is the better number.** A single value
  equal to each need's base rate scores ECE 0.0000; the model scores 0.001-0.008. Suppressing
  that would make the calibration screen say more than it can support, so `discrimination`
  goes out with it -- within-day AUC, pooled AUC, PR-AUC and lift, which is what actually
  separates Leeward from a predictor that has never met anyone. TRIPOD+AI asks for all three
  legs; this payload now carries them.
- `ablations` is read from `report/ablations.json`, which `make ablate` writes. An empty
  list still means "not run" -- six models over the real cohort is half a minute, too slow
  to sit inside a target that is re-run every few edits, and a stale table would be worse
  than a missing one. Run `make ablate` after `make score`; `make report` picks it up.
- **The fairness table goes in whole, flagged rows included.** `fairness_failed` is the
  audit's own verdict. Nothing here filters it.

    make report        # -> report/report.json + the CSVs and offline charts
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from leeward import schema
from leeward.api.schemas import ReportResponse
from leeward.eval import calibration as cal
from leeward.eval import decision_quality as dq
from leeward.eval import discrimination as disc
from leeward.eval import fairness as fair
from leeward.eval import recovery as rec

REPORT = schema.ROOT / "report"


def load_ablations(out_dir: Path = REPORT) -> list[dict]:
    """`AblationRow`s from the last `make ablate`, or `[]` if it has not been run.

    Cached rather than computed here: `ablate.run` re-scores the window once per block, and
    `make report` is run far more often than the ablations change.
    """
    path = out_dir / "ablations.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["ablations"]


def model_rung(scores: pl.DataFrame) -> int:
    """The one rung these scores were produced at."""
    rungs = scores["model_rung"].unique().to_list()
    if len(rungs) != 1:
        raise ValueError(f"scores mix model rungs {sorted(rungs)}; that cannot be explained "
                         "on stage. Re-score with one rung.")
    return int(rungs[0])


@dataclass(frozen=True)
class Report:
    """The JSON the API serves, plus the frames the CSVs and charts are written from."""
    payload: dict
    calibration: pl.DataFrame
    discrimination: pl.DataFrame
    recovery: pl.DataFrame
    fairness: pl.DataFrame
    decision_quality: pl.DataFrame

    def write(self, out_dir: Path = REPORT) -> list[Path]:
        """report.json and every side table. Returns what it wrote, in order."""
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "report.json"
        path.write_text(json.dumps(self.payload, indent=2) + "\n", encoding="utf-8")
        written = [path]
        written += list(cal.write_outputs(self.calibration, out_dir))
        written += list(disc.write_outputs(self.discrimination, out_dir))
        written += list(rec.write_outputs(self.recovery, out_dir))
        written += list(fair.write_outputs(self.fairness, out_dir))
        written += list(dq.write_outputs(self.decision_quality, out_dir))
        return written


def assemble(*, scores: pl.DataFrame, outcomes: pl.DataFrame, cohort: pl.DataFrame,
             dates: Sequence[date] | None = None, ks: Sequence[int] = dq.KS,
             n_draws: int = rec.N_DRAWS, seed: int = rec.SEED,
             posterior: Path | None = rec.POSTERIOR,
             ablations: Sequence[dict] | None = None) -> Report:
    """Run the whole harness over one held-out window."""
    dates = cal.holdout_dates(scores, outcomes) if dates is None else list(dates)

    # Paired once and shared: calibration, its control and discrimination all want the same
    # 1.5M-row frame, and the join to outcomes is the expensive part of the harness.
    pairs = cal.paired(scores, outcomes, dates=dates)
    reliability = cal.reliability(pairs)
    discrimination = disc.discriminate(pairs)
    recovered = rec.run(n_draws=n_draws, seed=seed, posterior=posterior)
    # The audit's coverage column is measured against the calls the care team would actually
    # have made, so it needs the allocator run over the same window the rest of the report
    # scores. `decision_quality` runs it again below at three budgets; the duplication costs
    # a few seconds and keeps each module runnable on its own.
    audit = fair.audit(scores, outcomes, cohort, dates=dates,
                       calls=fair.budget_calls(scores, cohort, dates=dates))
    w, tau = dq.decision_weights()
    tidy = dq.evaluate(scores, cohort, outcomes, w=w, tau=tau, ks=ks, dates=dates)
    rhat_max, divergences = rec.diagnostics(posterior)

    payload = {
        "model_rung": model_rung(scores),
        "rhat_max": rhat_max,
        "divergences": divergences,
        "recovery": recovered.select(
            "parameter", "truth", "post_mean", "lo90", "hi90", "covered").to_dicts(),
        "recovery_coverage": rec.coverage(recovered),
        "calibration": reliability.select("need", "predicted", "observed", "n").to_dicts(),
        "ece_by_need": cal.ece(reliability),
        # The control goes out with the claim. It is better than the model's ECE, which is
        # the finding, and the screen shows it: see `DiscriminationRow` for what does separate
        # the model from a single number at the base rate.
        "constant_ece": cal.constant_ece(pairs),
        "discrimination": discrimination.to_dicts(),
        "ablations": list(load_ablations() if ablations is None else ablations),
        "decision_quality": dq.summarise(tidy).select("k", "strategy",
                                                      "harm_averted").to_dicts(),
        "fairness": audit.select(fair.REPORT_COLUMNS).to_dicts(),
        "fairness_failed": fair.failed(audit),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    ReportResponse.model_validate(payload)   # fail here, not in the route, if a shape drifts
    return Report(payload=payload, calibration=reliability, discrimination=discrimination,
                  recovery=recovered, fairness=audit, decision_quality=tidy)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=cal.HOLDOUT_DAYS,
                    help="length of the held-out window (default: the last 30 scored days)")
    ap.add_argument("--draws", type=int, default=rec.N_DRAWS)
    ap.add_argument("--seed", type=int, default=rec.SEED)
    args = ap.parse_args(argv)

    scores, outcomes = schema.read("scores"), schema.read("outcomes")
    cohort = schema.read("cohort")
    dates = cal.holdout_dates(scores, outcomes, n_days=args.days)
    report = assemble(scores=scores, outcomes=outcomes, cohort=cohort, dates=dates,
                      n_draws=args.draws, seed=args.seed)
    written = report.write()

    p = report.payload
    print(f"rung {p['model_rung']} · held out {dates[0]} .. {dates[-1]} "
          f"({len(dates)} days, {scores.filter(pl.col('date').is_in(dates)).height:,} scored rows)")
    if p["rhat_max"] is None:
        print("  no MCMC at this rung, so no r-hat and no divergence count")
    else:
        print(f"  r-hat max {p['rhat_max']:.3f} · {p['divergences']} divergences")
    for need, value in p["ece_by_need"].items():
        control = p["constant_ece"].get(need, 0.0)
        print(f"  ECE {need:14s} {value:.4f}"
              f"{'' if value < cal.ECE_BAR else '   over the SPEC §11 bar'}"
              f"   · a constant at the base rate scores {control:.4f}")
    print("  ECE cannot separate the model from that constant at these base rates. "
          "Discrimination can:")
    for row in p["discrimination"]:
        within, pooled = row["within_day_auc"], row["pooled_auc"]
        if within is None or pooled is None:
            print(f"  AUC {row['need']:14s} not measurable: "
                  f"{row['n_events']} events in {row['n']:,} rows")
            continue
        print(f"  AUC {row['need']:14s} within-day {within:.3f}  (pooled {pooled:.3f}, "
              f"PR-AUC {row['pr_auc']:.3f}, top 1% of the day {row['lift_at_1pct']:.1f}x)")
    n_ablations = len(p["ablations"])
    print(f"  ablations {n_ablations} block(s) from report/ablations.json" if n_ablations
          else "  ablations not run -- `make ablate` fills the card on the Model report screen")
    kind = "prior coverage" if report.recovery["source"][0] == "prior" else "recovery"
    print(f"  {kind} {p['recovery_coverage']:.1%} of {len(p['recovery'])} parameters "
          f"(bar {rec.COVERAGE_BAR:.0%})")
    flagged = report.fairness.filter(pl.col("flagged"))
    if flagged.height:
        print(f"\nFAIRNESS AUDIT FAILED: {flagged.height} group(s) over the "
              f"{fair.FNR_GAP:.0%} relative FNR gap. Shown, never suppressed:")
        with pl.Config(tbl_rows=flagged.height + 1, tbl_width_chars=120):
            print(flagged.select("stratum", "group", "n", "n_events", "fnr",
                                 "fnr_ratio_to_cohort"))
    else:
        print(f"  fairness: no group over the {fair.FNR_GAP:.0%} relative FNR gap "
              f"({report.fairness.height} groups audited)")
    # The flag count is half the audit. Whether the bar could have been cleared at all, and
    # which groups were reached *more* than the cohort, are the other half and print either
    # way -- a report that says only "0 flagged" has not said what it found.
    for line in fair.ceiling_note(report.fairness).split(". "):
        print(f"  {line.strip().rstrip('.')}.")
    more = report.fairness.filter(pl.col("direction") == fair.REACHED_MORE)
    if more.height:
        top = more.sort("reach_ratio_to_cohort", descending=True).head(4)
        print(f"  {more.height} group(s) reached MORE than the cohort: " + ", ".join(
            f"{r['stratum']}:{r['group']} {r['reach_ratio_to_cohort']:.2f}x"
            for r in top.iter_rows(named=True)))
    print("wrote " + ", ".join(str(w.relative_to(schema.ROOT)) for w in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

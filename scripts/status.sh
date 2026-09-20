#!/usr/bin/env bash
# Where the build is, without opening a single source file.
# Run this instead of reading code. It is the async standup.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
g() { printf "\033[32m%s\033[0m" "$1"; }; r() { printf "\033[31m%s\033[0m" "$1"; }
d() { printf "\033[2m%s\033[0m" "$1"; }

echo
echo "LEEWARD STATUS  ·  $(date '+%a %H:%M')"
echo "──────────────────────────────────────────────────────────────"

echo
echo "MODULES"
while read -r label path; do
  if [ -e "$path" ]; then printf "  %s  %-34s" "$(g '●')" "$label"; d "$path"; echo
  else printf "  %s  %-34s" "$(r '○')" "$label"; d "$path"; echo; fi
done <<'MODS'
contracts·schema leeward/schema.py
contracts·api leeward/api/schemas.py
ingest·hazards leeward/ingest/hazards.py
cohort·build leeward/cohort/build.py
cohort·medications leeward/cohort/medications.py
cohort·simulate leeward/cohort/simulate.py
model·design leeward/model/design.py
model·score_prior leeward/model/score_prior.py
model·hazard(NUTS) leeward/model/hazard.py
model·fit leeward/model/fit.py
model·score leeward/model/score.py
decision·eha leeward/decision/eha.py
decision·allocate leeward/decision/allocate.py
outreach·messages leeward/outreach/messages.py
api·main leeward/api/main.py
eval·calibration leeward/eval/calibration.py
eval·fairness leeward/eval/fairness.py
eval·decision_quality leeward/eval/decision_quality.py
ui·CareTeam ui/src/screens/CareTeam.tsx
ui·VeteranCard ui/src/screens/VeteranCard.tsx
ui·Report ui/src/screens/Report.tsx
MODS

echo
echo "DATA TABLES"
for t in cohort hazards site_status outcomes scores actions outcome_log; do
  f="data/$t.parquet"
  if [ -f "$f" ]; then
    rows=$($PY -c "import polars as pl;print(f'{pl.read_parquet(\"$f\").height:,}')" 2>/dev/null || echo "?")
    printf "  %s  %-16s %10s rows\n" "$(g '●')" "$t" "$rows"
  else printf "  %s  %-16s %10s\n" "$(r '○')" "$t" "—"; fi
done

echo
echo "MODEL RUNG"
if [ -f data/scores.parquet ]; then
  $PY - <<'PYEOF' 2>/dev/null || echo "  unknown"
import polars as pl
s = pl.read_parquet("data/scores.parquet")
rung = s["model_rung"].unique().to_list()
names = {0: "prior-only (no MCMC)", 1: "pooled NUTS", 2: "interactions + SiteDown",
         3: "ICAR + latent dose"}
for x in rung:
    print(f"  rung {x} — {names.get(x, '?')}")
PYEOF
else echo "  no scores yet"; fi
[ -f data/posterior.nc ] && echo "  posterior.nc present" || echo "  posterior.nc absent (rung 0 needs none)"
if [ -f report/fit.json ]; then
  $PY - <<'PYEOF' 2>/dev/null || echo "  fit.json unreadable"
import json
f = json.load(open("report/fit.json"))
d = f["data"]
print(f"  last fit: rung {f['model_rung']}, {d['veteran_days']:,} veteran-days in "
      f"{d['cells']:,} cells ({d['reduction']}x), {d['days']} days to {d['last_date']}")
print(f"  r-hat max {f['rhat_max']:.4f} · {f['divergences']} divergences · "
      f"ESS min {f['ess_bulk_min']:,.0f} bulk / {f['ess_tail_min']:,.0f} tail · "
      f"{f['runtime_s']}s")
print(f"  {len(f['fitted_terms'])} terms fitted, {len(f['carried_terms'])} carried at "
      f"their priors and not in the prediction")
PYEOF
fi

echo
echo "GATE"
if .venv/bin/pytest -p no:cacheprovider --tb=line >/tmp/lw_test.txt 2>&1; then
  echo "  $(g '●') make check is GREEN — $(grep -oE '[0-9]+ passed[^=]*' /tmp/lw_test.txt | tail -1 | xargs)"
else
  echo "  $(r '●') make check is RED"
  grep -E "^FAILED" /tmp/lw_test.txt | sed 's/^/     /' | head -8
fi

echo
echo "GUARDRAILS WAITING (this is the to-do list)"
.venv/bin/pytest -q -rs -p no:cacheprovider 2>/dev/null \
  | grep -E '^SKIPPED' | sed 's/SKIPPED \[[0-9]*\] /  · /' | head -12 \
  || echo "  none"

echo
echo "LANES"
git worktree list 2>/dev/null | sed 's/^/  /' | head -8
echo
echo "RECENT"
git log --oneline -6 | sed 's/^/  /'
echo

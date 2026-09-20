#!/usr/bin/env bash
# Boot the API and hit every route. This is the demo acceptance test.
# It is what "the demo works" means, in a form nobody has to eyeball.
set -uo pipefail
cd "$(dirname "$0")/.."
PORT=${PORT:-8765}
PY=.venv/bin/python
FAIL=0

if [ ! -f leeward/api/main.py ]; then
  echo "leeward/api/main.py not built yet — smoke test skipped"; exit 0
fi

# No network. If a route reaches out, it fails here rather than on stage.
export HTTP_PROXY=http://127.0.0.1:1 HTTPS_PROXY=http://127.0.0.1:1 NO_PROXY=127.0.0.1,localhost

$PY -m uvicorn leeward.api.main:app --port "$PORT" --log-level warning &
API=$!
trap 'kill $API 2>/dev/null' EXIT

for i in $(seq 1 40); do
  curl -sf "http://127.0.0.1:$PORT/openapi.json" >/dev/null 2>&1 && break
  sleep 0.5
done

hit() {  # hit <method> <path> <jq-ish python check>
  local method=$1 path=$2 check=${3:-}
  local body; body=$(curl -sf -X "$method" -H 'content-type: application/json' \
    ${4:+-d "$4"} "http://127.0.0.1:$PORT$path" 2>/dev/null)
  if [ -z "$body" ]; then echo "  ✗ $method $path — no response"; FAIL=1; return; fi
  if [ -n "$check" ]; then
    echo "$body" | $PY -c "import sys,json; d=json.load(sys.stdin); assert $check, 'check failed'" \
      2>/dev/null && echo "  ✓ $method $path" || { echo "  ✗ $method $path — $check"; FAIL=1; }
  else echo "  ✓ $method $path"; fi
}

DAY=$($PY -c "import polars as pl;print(pl.read_parquet('data/actions.parquet')['date'][0])")
VET=$($PY -c "import polars as pl;print(pl.read_parquet('data/cohort.parquet')['veteran_id'][0])")

echo "smoke test against http://127.0.0.1:$PORT (network blocked)"
# No `day`: the board opens where something is happening. A quiet headline here means
# the demo opens on an empty ribbon, which is the one screen we cannot show first.
hit GET  "/forecast?scenario=sandy_then_heat" \
     "len(d['zips'])>100 and 'No active alerts' not in d['headline']"
hit GET  "/forecast?scenario=sandy_then_heat&day=0" "len(d['zips'])>100"
hit GET  "/scores?date=$DAY&need=heat"              "len(d['zips'])>0"
hit GET  "/veteran/$VET?date=$DAY"                  "len(d['needs'])==5"
hit POST "/actions" "len(d['actions'])>0 and d['total_eha']>0" \
     "{\"date\":\"$DAY\",\"capacity\":{\"call\":40,\"refill\":200,\"ride\":15,\"booking\":20,\"evac\":8,\"partner_slot\":10,\"pharmacist_slot\":12,\"va_fill\":30,\"free\":10000}}"
hit GET  "/report"                                  "'model_rung' in d"

echo
[ $FAIL -eq 0 ] && echo "SMOKE PASSED — every route answers offline" \
                || echo "SMOKE FAILED"
exit $FAIL

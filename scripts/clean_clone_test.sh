#!/usr/bin/env bash
# Prove a fresh clone boots offline AND serves the UI: clone, install, fixtures, boot, then fetch
# the page and one real route -- and print the wall-clock seconds. Run this Sunday morning, not
# Sunday afternoon.
#
#   bash scripts/clean_clone_test.sh               # the demo path, timed against BUDGET_S (60)
#   bash scripts/clean_clone_test.sh --with-tests  # + pytest in the clone, timed separately
#
# It clones what is COMMITTED. An uncommitted edit to ui/src, or a ui/dist you rebuilt but did not
# commit, is not in the clone -- which is the point: this is what a laptop with no node_modules
# and no wifi will get.
#
# The network is blocked with the proxy variables, as smoke_demo.sh does, from the moment the
# clone exists: the install must come out of uv's wheel cache (`--offline`), and any fetch the
# API makes fails here instead of on stage. It cannot stop a browser reaching out, so once,
# before the demo, run it with the wifi actually off.
set -uo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
BUDGET_S=${BUDGET_S:-60}
PORT=${PORT:-8766}                      # not 8000 (the demo) or 8765 (smoke_demo.sh)
WITH_TESTS=0
[ "${1:-}" = "--with-tests" ] && WITH_TESTS=1

now()     { perl -MTime::HiRes=time -e 'printf "%.2f", time'; }
elapsed() { perl -e 'printf "%.1f", $ARGV[0] - $ARGV[1]' "$1" "$2"; }

TMP=$(mktemp -d)
LOG="$TMP/api.log"
API=""
trap '[ -n "$API" ] && kill "$API" 2>/dev/null; rm -rf "$TMP"' EXIT

fail() {
  echo >&2; echo "CLEAN CLONE FAILED: $*" >&2
  [ -s "$LOG" ] && { echo "--- API log:" >&2; tail -20 "$LOG" >&2; }
  exit 1
}

# Something already on the port would answer our curls and make a dead build look alive.
if curl -s -o /dev/null "http://127.0.0.1:$PORT/"; then
  fail "port $PORT is already answering (a stale server?). Kill it, or set PORT=."
fi

T0=$(now); LAST=$T0
declare -a PHASES=()
phase() { local t; t=$(now); PHASES+=("$1 $(elapsed "$t" "$LAST")s"); LAST=$t; }

echo "cloning $SRC @ $(git -C "$SRC" rev-parse --short HEAD) into $TMP"
DIRTY=$(git -C "$SRC" status --porcelain | wc -l | tr -d ' ')
[ "$DIRTY" != "0" ] && echo "  note: $DIRTY uncommitted path(s) in $SRC are not in the clone"
git clone -q "$SRC" "$TMP/leeward" || fail "git clone"
cd "$TMP/leeward" || exit 1
phase clone

export HTTP_PROXY=http://127.0.0.1:1 HTTPS_PROXY=http://127.0.0.1:1 ALL_PROXY=http://127.0.0.1:1
export http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY all_proxy=$ALL_PROXY
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

# The committed bundle is the whole reason a clone can serve a page without npm. Check it first:
# it costs milliseconds and a stale one should not be discovered after a 30 s install.
python3 scripts/ui_dist.py check || fail "ui/dist is missing or stale relative to ui/src; run 'make ui' and commit ui/dist"
phase dist-check

EXTRAS=""; [ "$WITH_TESTS" = 1 ] && EXTRAS="[dev]"
if command -v uv >/dev/null 2>&1; then
  uv venv --offline --python 3.11 -q .venv \
    && uv pip install --offline -q --python .venv/bin/python -e ".$EXTRAS"
else
  python3.11 -m venv .venv && .venv/bin/python -m pip install -q --retries 0 --timeout 5 -e ".$EXTRAS"
fi || fail "install failed with the network blocked. A machine that has never run 'make setup' has no wheel cache to install from; run it once while online."
phase install

.venv/bin/python scripts/make_fixtures.py >/dev/null || fail "make_fixtures.py"
phase fixtures

.venv/bin/python -m uvicorn leeward.api.main:app --port "$PORT" --log-level warning >"$LOG" 2>&1 &
API=$!
disown "$API"                           # so bash does not print "Terminated" when the trap reaps it
UP=0
for _ in $(seq 1 240); do
  kill -0 "$API" 2>/dev/null || fail "uvicorn exited while booting"
  curl -sf "http://127.0.0.1:$PORT/openapi.json" >/dev/null 2>&1 && { UP=1; break; }
  sleep 0.25
done
[ "$UP" = 1 ] || fail "the API did not answer within 60 s"
phase boot

URL="http://127.0.0.1:$PORT"
PAGE=$(curl -sf "$URL/") || fail "GET / did not answer: the UI is not being served"
echo "$PAGE" | grep -q 'id="root"' || fail "GET / answered, but it is not the app: no id=\"root\" in the page"
JS=$(echo "$PAGE" | grep -oE 'src="/assets/[^"]+\.js"' | head -1 | sed -e 's/^src="//' -e 's/"$//')
[ -n "$JS" ] || fail "the page has no /assets/*.js script"
curl -sf "$URL$JS" -o "$TMP/app.js" && [ -s "$TMP/app.js" ] || fail "the page's script $JS is not served"
# The route the UI actually calls: /api/forecast, the prefix the Vite proxy strips in dev.
curl -sf "$URL/api/forecast?scenario=sandy_then_heat" \
  | .venv/bin/python -c "import sys,json; d=json.load(sys.stdin); assert len(d['zips'])>100, len(d['zips'])" \
  || fail "GET /api/forecast did not return a forecast"
phase page+route

TOTAL=$(elapsed "$(now)" "$T0")
echo
echo "  served / (app root), $JS, and /api/forecast -- network blocked"
for p in "${PHASES[@]}"; do printf '    %-12s %7s\n' "${p% *}" "${p##* }"; done
if ! perl -e 'exit($ARGV[0] <= $ARGV[1] ? 0 : 1)' "$TOTAL" "$BUDGET_S"; then
  fail "took ${TOTAL}s; the budget is ${BUDGET_S}s (BUDGET_S=)"
fi
echo "CLEAN CLONE OK: booted and served the UI in ${TOTAL}s (budget ${BUDGET_S}s)"

if [ "$WITH_TESTS" = 1 ]; then
  T1=$(now)
  .venv/bin/pytest -q -p no:cacheprovider || fail "tests failed on a clean clone"
  echo "clean clone pytest green in $(elapsed "$(now)" "$T1")s (not counted against the budget)"
fi

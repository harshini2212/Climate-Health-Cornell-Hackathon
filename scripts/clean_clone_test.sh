#!/usr/bin/env bash
# Prove a fresh clone boots offline. Run this Sunday morning, not Sunday afternoon.
set -uo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
echo "cloning into $TMP"
git clone -q "$SRC" "$TMP/leeward" || exit 1
cd "$TMP/leeward"
START=$(date +%s)
uv venv --python 3.11 .venv -q 2>/dev/null || python3.11 -m venv .venv
uv pip install -q -e ".[dev]" 2>/dev/null || .venv/bin/python -m pip install -q -e ".[dev]"
.venv/bin/python scripts/make_fixtures.py >/dev/null || { echo "fixtures failed"; exit 1; }
.venv/bin/pytest -q -p no:cacheprovider || { echo "tests failed on a clean clone"; exit 1; }
echo "clean clone green in $(( $(date +%s) - START ))s"

#!/usr/bin/env bash
# Print prompt N from docs/PROMPTS.md, ready to paste into a Claude Code session.
#   bash scripts/prompt.sh 4            # print it
#   bash scripts/prompt.sh 4 | pbcopy   # straight to the clipboard
#   bash scripts/prompt.sh              # list them all
set -euo pipefail
cd "$(dirname "$0")/.."
F=docs/PROMPTS.md

if [ $# -eq 0 ]; then
  echo "Prompts in $F — one per terminal, six per round:"
  grep -nE "^### [0-9]+ · " "$F" | sed -E 's/^[0-9]+:### /  /'
  echo
  echo "  bash scripts/prompt.sh <N> | pbcopy"
  exit 0
fi

N=$1
START=$(grep -nE "^### $N · " "$F" | head -1 | cut -d: -f1)
[ -n "${START:-}" ] || { echo "no prompt $N; run with no argument to list them" >&2; exit 1; }
END=$(awk -v s="$START" 'NR>s && /^(### |## )/ {print NR; exit}' "$F")
[ -n "${END:-}" ] || END=$(wc -l < "$F")

# Strip the heading and the leading "> " so it pastes as plain text.
sed -n "$((START+1)),$((END-1))p" "$F" | sed -E 's/^> ?//'

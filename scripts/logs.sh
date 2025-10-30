#!/usr/bin/env bash
set -euo pipefail

TAIL=50
if [[ $# -gt 0 ]]; then
  TAIL="$1"
fi

source "$(dirname "$0")/common.sh"
LOG_FILE="$REPO_ROOT/artifacts/logs/genie_lamp.log.jsonl"
if [[ ! -f "$LOG_FILE" ]]; then
  echo "No structured log file found at $LOG_FILE yet." >&2
  exit 0
fi

echo "[+] Last $TAIL log lines"
tail -n "$TAIL" "$LOG_FILE"

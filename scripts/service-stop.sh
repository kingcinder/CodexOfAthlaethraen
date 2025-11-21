#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

PID_PATH="$REPO_ROOT/artifacts/run/uvicorn.pid"
if [[ ! -f "$PID_PATH" ]]; then
  exit 0
fi

PID="$(cat "$PID_PATH")"
if kill -0 "$PID" >/dev/null 2>&1; then
  echo "[+] Stopping Genie Lamp process (PID $PID)"
  kill "$PID" || true
  wait "$PID" 2>/dev/null || true
fi
rm -f "$PID_PATH"

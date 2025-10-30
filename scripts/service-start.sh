#!/usr/bin/env bash
set -euo pipefail

USE_GPU=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --use-gpu)
      USE_GPU=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

source "$(dirname "$0")/common.sh"
import_dotenv "$REPO_ROOT/.env"
export GENIE_LAMP_HOME="$REPO_ROOT"

PID_PATH="$REPO_ROOT/artifacts/run/uvicorn.pid"
if [[ -f "$PID_PATH" ]]; then
  if kill -0 "$(cat "$PID_PATH")" >/dev/null 2>&1; then
    echo "[+] Existing Genie Lamp process detected ($(cat "$PID_PATH")), stopping"
    "$REPO_ROOT/scripts/service-stop.sh"
  fi
fi

HOST="$(get_service_host)"
PORT="$(get_service_port)"
STDOUT_LOG="$REPO_ROOT/artifacts/logs/uvicorn.stdout.log"
STDERR_LOG="$REPO_ROOT/artifacts/logs/uvicorn.stderr.log"
ensure_dir "$(dirname "$STDOUT_LOG")"

ARGS=("$VENV_PYTHON" -m genie_lamp serve --host "$HOST" --port "$PORT")
if $USE_GPU; then
  export GENIE_LAMP_USE_GPU=1
  ARGS+=(--use-gpu)
fi

echo "[+] Starting Genie Lamp service on ${HOST}:${PORT}"
(
  cd "$REPO_ROOT"
  nohup "${ARGS[@]}" >"$STDOUT_LOG" 2>"$STDERR_LOG" &
  echo $! >"$PID_PATH"
)

sleep 3

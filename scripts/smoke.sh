#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
import_dotenv "$REPO_ROOT/.env"

HOST="$(get_service_host)"
PORT="$(get_service_port)"
DIAG_URL="http://${HOST}:${PORT}/diag"

echo "[+] Fetching diagnostics from $DIAG_URL"
DIAG_JSON="$(curl --fail --silent --max-time 10 "$DIAG_URL")"
if [[ -z "$DIAG_JSON" ]]; then
  echo "Diagnostics endpoint returned no data" >&2
  exit 1
fi
echo "[+] Diagnostic payload: $DIAG_JSON"

echo "[+] Running smoke tests"
(
  cd "$REPO_ROOT"
  "$VENV_PYTHON" -m pytest app/tests/smoke -q
)

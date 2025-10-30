#!/usr/bin/env bash
set -euo pipefail

RETRIES=15
DELAY=2

source "$(dirname "$0")/common.sh"
import_dotenv "$REPO_ROOT/.env"

HOST="$(get_service_host)"
PORT="$(get_service_port)"
HEALTH_URL="http://${HOST}:${PORT}/health"

echo "[+] Probing $HEALTH_URL"
for ((i=0; i<RETRIES; i++)); do
  if curl --fail --silent --max-time 5 "$HEALTH_URL" | grep -q '"ok": true'; then
    echo "[+] Health check passed"
    exit 0
  fi
  sleep "$DELAY"
done

echo "Health check failed after ${RETRIES} attempts" >&2
exit 1

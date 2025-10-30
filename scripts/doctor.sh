#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
import_dotenv "$REPO_ROOT/.env"

echo "[+] Verifying Python interpreter"
PYTHON_BIN="$(command -v python)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3.11.x is required" >&2
  exit 1
fi
PY_VERSION="$(python -c 'import sys; print(sys.version)')"
if [[ $PY_VERSION != 3.11* ]]; then
  echo "Python 3.11.x is required. Found: $PY_VERSION" >&2
  exit 1
fi
echo "[+] Python $PY_VERSION detected"

echo "[+] Ensuring curl availability"
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for health checks" >&2
  exit 1
fi

echo "[+] Environment check completed"

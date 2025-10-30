#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
import_dotenv "$REPO_ROOT/.env"

ensure_dir "$REPO_ROOT/artifacts/dist"

echo "[+] Building wheel artifact"
(
  cd "$REPO_ROOT"
  "$VENV_PYTHON" -m pip wheel . -w artifacts/dist --no-deps
)

echo "[+] Wheel available under artifacts/dist"

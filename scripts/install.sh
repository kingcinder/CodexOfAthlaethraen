#!/usr/bin/env bash
set -euo pipefail

ALLOW_DOWNLOAD=false
USE_GPU=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-download-models)
      ALLOW_DOWNLOAD=true
      shift
      ;;
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

echo "[+] Preparing Python virtual environment"
if [[ ! -d "$VENV_PATH" ]]; then
  python -m venv "$VENV_PATH"
fi

echo "[+] Upgrading packaging tools"
"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

echo "[+] Installing Genie Lamp in editable mode"
(
  cd "$REPO_ROOT"
  "$VENV_PYTHON" -m pip install -e .
  "$VENV_PYTHON" -m pip install pytest==8.3.4
)

if $USE_GPU; then
  echo "[+] GPU flag detected. Install CUDA-enabled torch manually if supported:"
  echo "    $VENV_PYTHON -m pip install torch==2.9.0+cu121 --index-url https://download.pytorch.org/whl/cu121"
fi

if $ALLOW_DOWNLOAD; then
  "$REPO_ROOT/scripts/fetch-model.sh" --allow-download
else
  "$REPO_ROOT/scripts/fetch-model.sh"
fi

ensure_dir "$REPO_ROOT/artifacts/logs"
ensure_dir "$REPO_ROOT/artifacts/models"
ensure_dir "$REPO_ROOT/artifacts/run"

echo "[+] Installation completed"

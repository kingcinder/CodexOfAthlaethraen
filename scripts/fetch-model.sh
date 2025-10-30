#!/usr/bin/env bash
set -euo pipefail

ALLOW_DOWNLOAD=false
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--allow-download" ]]; then
    ALLOW_DOWNLOAD=true
  else
    echo "Unknown option: $1" >&2
    exit 1
  fi
fi

source "$(dirname "$0")/common.sh"
import_dotenv "$REPO_ROOT/.env"

MODEL_ROOT="$REPO_ROOT/artifacts/models/sentence-transformers"
TARGET_DIR="$MODEL_ROOT/all-MiniLM-L6-v2"
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
ensure_dir "$MODEL_ROOT"

if [[ -f "$TARGET_DIR/config.json" ]]; then
  echo "[+] Sentence-Transformer cache already present at $TARGET_DIR"
  exit 0
fi

if ! $ALLOW_DOWNLOAD; then
  echo "Model artifacts missing at $TARGET_DIR." >&2
  echo "Place the extracted $MODEL_NAME directory under artifacts/models/sentence-transformers or rerun with --allow-download." >&2
  exit 0
fi

echo "[+] Downloading $MODEL_NAME to $MODEL_ROOT"
(
  cd "$REPO_ROOT"
  "$VENV_PYTHON" - <<'PY'
from pathlib import Path
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-MiniLM-L6-v2"
cache = Path("artifacts/models/sentence-transformers")
SentenceTransformer(model_name, cache_folder=str(cache))
PY
)

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$REPO_ROOT/.venv"
VENV_PYTHON="$VENV_PATH/bin/python"
LOG_DIR="$REPO_ROOT/artifacts/logs"
RUN_DIR="$REPO_ROOT/artifacts/run"

ensure_dir() {
  mkdir -p "$1"
}

import_dotenv() {
  local env_file="$1"
  if [[ ! -f "$env_file" ]]; then
    return
  fi
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || ${line:0:1} == "#" ]] && continue
    if [[ "$line" == *"="* ]]; then
      local key="${line%%=*}"
      local value="${line#*=}"
      export "${key}"="${value}"
    fi
  done <"$env_file"
}

get_service_host() {
  if [[ -n "${GENIE_LAMP_HOST:-}" ]]; then
    printf '%s' "$GENIE_LAMP_HOST"
  else
    printf '127.0.0.1'
  fi
}

get_service_port() {
  if [[ -n "${GENIE_LAMP_PORT:-}" ]]; then
    printf '%s' "$GENIE_LAMP_PORT"
  else
    printf '7860'
  fi
}

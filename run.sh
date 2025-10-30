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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"
LOG_DIR="$REPO_ROOT/artifacts/logs"
ensure_dir "$LOG_DIR"
RUN_LOG="$LOG_DIR/run.sh.log"

trap "$REPO_ROOT/scripts/service-stop.sh" EXIT

{
  echo "[+] Step 1/6: Environment diagnostics"
  "$REPO_ROOT/scripts/doctor.sh"

  echo "[+] Step 2/6: Installation"
  INSTALL_ARGS=()
  $ALLOW_DOWNLOAD && INSTALL_ARGS+=("--allow-download-models")
  $USE_GPU && INSTALL_ARGS+=("--use-gpu")
  "$REPO_ROOT/scripts/install.sh" "${INSTALL_ARGS[@]}"

  echo "[+] Step 3/6: Build artifacts"
  "$REPO_ROOT/scripts/build.sh"

  echo "[+] Step 4/6: Launch service"
  SERVICE_ARGS=()
  $USE_GPU && SERVICE_ARGS+=("--use-gpu")
  "$REPO_ROOT/scripts/service-start.sh" "${SERVICE_ARGS[@]}"

  echo "[+] Step 5/6: Health verification"
  "$REPO_ROOT/scripts/health.sh"

  echo "[+] Step 6/6: Smoke tests"
  "$REPO_ROOT/scripts/smoke.sh"

  echo "[+] Collecting tail of structured logs"
  "$REPO_ROOT/scripts/logs.sh" 20
} | tee -a "$RUN_LOG"

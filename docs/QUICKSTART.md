# Genie Lamp Quickstart

## Prerequisites
- Windows 10/11 with PowerShell 5.1+ or Windows Terminal
- Python 3.11.x on PATH (`python --version`)
- `curl` for local health checks
- Optional: Docker Engine 24+ for container builds

## First Run (Windows)
```powershell
# From the repository root
./run.ps1 -AllowNetworkModels
```

## First Run (Linux)
```bash
# From the repository root
./run.sh --allow-download-models
```

Each run performs:
1. Environment diagnostics (Python and curl checks)
2. Editable install inside `.venv`
3. Deterministic wheel build in `artifacts/dist`
4. Service launch on `http://127.0.0.1:7860`
5. Health and diagnostics probes
6. Smoke tests under `app/tests/smoke`
7. Log tail written to `artifacts/logs`

## Configuration
- Defaults live in `app/config/default.yaml`
- Override via `.env` (copy `.env.example`) or `GENIE_LAMP_CONFIG`
- Structured logs land in `artifacts/logs/genie_lamp.log.jsonl`
- Models cache under `artifacts/models`

## Offline Model Preparation
- Place `sentence-transformers/all-MiniLM-L6-v2` under `artifacts/models/sentence-transformers`
- Or execute `./scripts/fetch-model.ps1 -AllowNetwork` (PowerShell) / `./scripts/fetch-model.sh --allow-download` (Linux)

## Container Build
```bash
# Build image
DOCKER_BUILDKIT=1 docker build -f containers/Dockerfile -t genie-lamp:local .

# Run container
docker run --rm -p 7860:7860 -v "$PWD/artifacts":/app/artifacts genie-lamp:local
```

## Verification Endpoints
- `GET /health` → `{"ok": true}`
- `GET /diag` → Torch/Transformers versions and CUDA flag

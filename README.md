# Codex of Athlaethraen

Genie Lamp now ships as a Windows-first, offline-friendly FastAPI service with reproducible build tooling.

## Quickstart
- **Windows:** `./run.ps1`
- **Linux:** `./run.sh`

Both scripts execute environment diagnostics, editable installation, wheel build, service launch, health checks, smoke tests, and log tailing. Structured logs land in `artifacts/logs/` and model caches in `artifacts/models/`.

See [`docs/QUICKSTART.md`](docs/QUICKSTART.md) for detailed instructions, container usage, and configuration guidance.

## Repository Layout
```
/
├─ app/                  # Source, configs, tests
├─ artifacts/            # Logs, models, runtime state
├─ containers/           # Docker assets
├─ docs/                 # Operational documentation
├─ scripts/              # Automation (PowerShell & Bash)
├─ tools/                # Utilities
└─ run.(ps1|sh)          # Single-command orchestration
```

## Key Automation Assets
- **Bootstrap & install:** `scripts/bootstrap.ps1` (Windows) / `scripts/install.sh` (Linux) — create or update `.venv` and perform editable installs.
- **One-step repair:** `scripts/repair-venv.ps1` — deletes and recreates `.venv`, runs `ensurepip`, and reinstalls the package with pinned dependencies while logging to `artifacts/logs/repair-*.log`.
- **Service orchestration:** `run.ps1` / `run.sh` — orchestrate diagnostics, install, build, launch, health checks, smoke tests, and log tailing in a single command.
- **Model cache helper:** `scripts/fetch-model.ps1` / `scripts/fetch-model.sh` — populate `artifacts/models/sentence-transformers/all-MiniLM-L6-v2/` when offline caches are missing.
- **Operational docs:** `docs/QUICKSTART.md`, `docs/CONFIG.md`, and `docs/TROUBLESHOOTING.md` — detail install, configuration precedence, reloader behaviour, and recovery steps.

## Legacy Glyphs
Historical manuscripts remain under `docs/super_saiyan_genie_lamp.md` and auxiliary projects like `the_assistant/` continue to ship with the repo.

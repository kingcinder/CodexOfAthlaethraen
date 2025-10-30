# Troubleshooting

## SentenceTransformer model missing offline
**Symptoms:** `/ready` returns `503` with `SentenceTransformer model not available offline` or the service crashes on startup.

**Resolution:**
1. Ensure the directory `artifacts/models/sentence-transformers/all-MiniLM-L6-v2/` contains the model files (`config.json`, `pytorch_model.bin`, etc.).
2. If you have network access, run `./scripts/fetch-model.ps1 -AllowNetwork` (Windows) or `./scripts/fetch-model.sh --allow-download` (Linux) after bootstrapping to populate the cache.
3. To permit the service to download models during startup, export `ALLOW_NETWORK=1` before running `./run.ps1` or `./scripts/run.ps1`.

## SQLite FTS5 unavailable on Windows
**Symptoms:** `/ready` returns `503` mentioning `SQLite FTS5 extension unavailable` or `/diag` reports `fts_available: false` with an `fts_error` hint.

**Resolution:**
1. Ensure your Python installation ships with SQLite compiled with FTS5 (Python 3.11 from python.org includes it). Windows Store Python builds may omit the module.
2. Re-run `python -m sqlite3` and execute `SELECT sqlite_compileoption_used('ENABLE_FTS5');` — it must return `1`.
3. If disabled, install a Python distribution with FTS5 support or rebuild SQLite with FTS5 enabled, then rerun `scripts/bootstrap.ps1` to recreate the virtual environment.

## Reloader & PYTHONPATH
**Symptoms:** Uvicorn reload subprocesses crash with `ModuleNotFoundError: genie_lamp` or hot reload fails when using `scripts/run.ps1`.

**Resolution:**
1. `scripts/run.ps1` sets `$env:PYTHONPATH = (Resolve-Path .).Path` before launching Uvicorn so the autoreloader inherits a stable import path to the repository root. Without this, Windows reload workers may not resolve package imports.
2. The reload subprocess inherits the environment and reuses the same interpreter, ensuring `genie_lamp` remains importable after code changes.
3. Verify the environment propagation with:
   ```powershell
   .\.venv\Scripts\python.exe -c "import os; print(os.environ.get('PYTHONPATH'))"
   ```
   The output should match the absolute repository path. If it is empty, rerun `scripts/run.ps1` from the repo root so the variable is set before launching the server.

## Virtual environment repair failures
**Symptoms:** `.venv` is corrupted, pip commands fail, or dependencies are missing even after rerunning `scripts/bootstrap.ps1`.

**Resolution:**
1. Execute `./scripts/repair-venv.ps1` to recreate the virtual environment from scratch. The script removes only `.venv`, reruns `ensurepip`, reinstalls the project in editable mode, and logs to `artifacts/logs/repair-*.log`.
2. Review the generated log for detailed errors. On success the script prints `PASS`; on failure it prints `FAIL` with next steps.
3. After a successful repair, restart the workflow with `./scripts/run.ps1` (dev) or `./run.ps1` to relaunch the service.

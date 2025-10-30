from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from genie_lamp import __version__
from genie_lamp.core import vector_store as vector_store_module
from genie_lamp.core.config import DEFAULT_CONFIG_PATH


def _sanitize_path(raw_path: str | None) -> str:
    if not raw_path:
        return ""
    try:
        resolved = Path(raw_path).expanduser().resolve()
    except Exception:
        return raw_path

    home_env = os.environ.get("GENIE_LAMP_HOME")
    if home_env:
        try:
            home_path = Path(home_env).expanduser().resolve()
            if resolved.is_relative_to(home_path):
                return str(Path("<GENIE_LAMP_HOME>") / resolved.relative_to(home_path))
        except Exception:
            pass
    return str(resolved)


def _ready_components(app: FastAPI) -> tuple[object, object]:
    agent = getattr(app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")
    memory = getattr(agent, "mem", None)
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory subsystem unavailable")
    vector = getattr(memory, "vs", None)
    if vector is None:
        raise HTTPException(status_code=503, detail="Vector store not initialised")
    return memory, vector


def _build_config_preview(cfg: dict) -> dict:
    def _copy_and_sanitize(section: dict | None, path_keys: set[str]) -> dict:
        if not section:
            return {}
        result: dict = {}
        for key, value in section.items():
            if isinstance(value, str) and key in path_keys:
                result[key] = _sanitize_path(value)
            else:
                result[key] = value
        return result

    retrieval = _copy_and_sanitize(cfg.get("retrieval"), {"cache_dir"})
    memory = _copy_and_sanitize(cfg.get("memory"), {"persist_path", "sqlite_path"})
    models = _copy_and_sanitize(cfg.get("models"), {"transformers_cache"})
    flags = cfg.get("flags", {}) or {}

    return {
        "retrieval": retrieval,
        "memory": memory,
        "models": models,
        "flags": flags,
    }


def register_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/ready")
    def ready():
        memory, vector = _ready_components(app)

        if not hasattr(vector, "encoder") or vector.encoder is None:
            raise HTTPException(status_code=503, detail="SentenceTransformer encoder not loaded")

        index_ready = getattr(vector, "index", None) is not None or hasattr(vector, "_mem")
        if not index_ready:
            raise HTTPException(status_code=503, detail="Vector index not initialised")

        hybrid = getattr(memory, "hybrid", None)
        if hybrid and getattr(hybrid, "fts_available", False) is False and memory.cfg.get("retrieval", {}).get("hybrid"):
            raise HTTPException(status_code=503, detail="SQLite FTS5 extension unavailable")

        return {"status": "ready"}

    @app.get("/version")
    def version():
        return {"name": "genie-lamp", "version": __version__}

    @app.get("/config/preview")
    def config_preview():
        cfg = getattr(app.state, "config", {})
        return _build_config_preview(cfg)

    @app.get("/diag")
    def diag():
        import platform
        import sys

        info = {"python": sys.version, "platform": platform.platform()}
        cfg = getattr(app.state, "config", {})
        info["config_path"] = str(DEFAULT_CONFIG_PATH)
        info["genie_lamp_home"] = os.environ.get("GENIE_LAMP_HOME", "")
        info["log_path"] = str(cfg.get("logging", {}).get("json_path", ""))

        memory = None
        vector = None
        try:
            memory, vector = _ready_components(app)
        except HTTPException:
            pass

        hybrid = getattr(memory, "hybrid", None) if memory else None

        info["faiss_present"] = bool(vector_store_module.faiss is not None)
        info["encoder_dim"] = getattr(vector, "dim", None) if vector else None
        retrieval_cfg = cfg.get("retrieval", {})
        info["retrieval"] = {
            "hybrid": bool(retrieval_cfg.get("hybrid")),
            "fts_available": bool(getattr(hybrid, "fts_available", False)),
        }
        if getattr(hybrid, "fts_error", None):
            info["retrieval"]["fts_error"] = str(hybrid.fts_error)
        if "use_fts5" in retrieval_cfg:
            info["retrieval"]["use_fts5"] = bool(retrieval_cfg.get("use_fts5"))
        memory_cfg = cfg.get("memory", {})
        info["memory"] = {
            "sqlite_path": _sanitize_path(memory_cfg.get("sqlite_path"))
        }

        try:
            import torch  # type: ignore

            info["torch"] = torch.__version__
            info["cuda"] = bool(
                getattr(torch.cuda, "is_available", lambda: False)()  # type: ignore[attr-defined]
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            info["torch_error"] = repr(exc)
        try:
            import transformers  # type: ignore

            info["transformers"] = transformers.__version__
        except Exception as exc:  # pragma: no cover - diagnostic path
            info["transformers_error"] = repr(exc)

        return info


__all__ = ["register_routes"]

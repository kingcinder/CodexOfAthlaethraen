"""Model loading utilities with offline-first behaviour."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from sentence_transformers import SentenceTransformer

from genie_lamp.core.config import PROJECT_ROOT

MODEL_SUBDIR = Path("artifacts") / "models" / "sentence-transformers"
MODEL_NAME_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"


def _collect_candidate_roots(cfg: Dict[str, Any]) -> Iterable[Path]:
    retrieval_cfg = cfg.get("retrieval", {})
    models_cfg = cfg.get("models", {})
    cache_dir_cfg = retrieval_cfg.get("cache_dir") or models_cfg.get("sentence_transformers_cache")

    if cache_dir_cfg:
        candidate = Path(cache_dir_cfg)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        yield candidate

    yield (PROJECT_ROOT / MODEL_SUBDIR).resolve()


def _find_local_model_dir(root: Path, model_name: str) -> Optional[Path]:
    potential_dirs = [root]
    model_parts = Path(model_name)
    potential_dirs.append(root / model_parts)
    potential_dirs.append(root / model_name.replace("/", os.sep))
    potential_dirs.append(root / model_name.split("/")[-1])

    for candidate in potential_dirs:
        config_file = candidate / "config.json"
        if config_file.exists():
            return candidate
    return None


def load_sentence_encoder(cfg: Dict[str, Any]) -> SentenceTransformer:
    """Load a sentence encoder preferring local artifacts."""

    retrieval_cfg = cfg.get("retrieval", {})
    memory_cfg = cfg.get("memory", {})
    model_name = retrieval_cfg.get("model", memory_cfg.get("embedder", MODEL_NAME_DEFAULT))

    candidate_roots = list(_collect_candidate_roots(cfg))
    for root in candidate_roots:
        root.mkdir(parents=True, exist_ok=True)
        local_dir = _find_local_model_dir(root, model_name)
        if local_dir is not None:
            return SentenceTransformer(str(local_dir))

    allow_network = os.environ.get("ALLOW_NETWORK", "0") == "1"
    if not allow_network:
        expected_root = candidate_roots[0] if candidate_roots else (PROJECT_ROOT / MODEL_SUBDIR).resolve()
        hints = [
            f"Expected to find '{model_name}' under {expected_root} with a config.json file.",
            "Populate the directory manually or rerun scripts/fetch-model.ps1 -AllowNetwork.",
            "To permit on-demand download, set ALLOW_NETWORK=1 before launching the service.",
        ]
        raise RuntimeError("\n".join(["SentenceTransformer model not available offline."] + hints))

    download_root = candidate_roots[0] if candidate_roots else (PROJECT_ROOT / MODEL_SUBDIR).resolve()
    download_root.mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(model_name, cache_folder=str(download_root))


__all__ = ["load_sentence_encoder"]

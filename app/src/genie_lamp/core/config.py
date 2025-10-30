from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency safeguard
    yaml = None  # type: ignore

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _detect_project_root() -> Path:
    env_home = os.environ.get("GENIE_LAMP_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    for candidate in [PACKAGE_ROOT] + list(PACKAGE_ROOT.parents):
        config_candidate = candidate / "config" / "default.yaml"
        if config_candidate.exists():
            return candidate.resolve()
    return Path.cwd().resolve()


PROJECT_ROOT = _detect_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
ENV_OVERRIDE_KEY = "GENIE_LAMP_OVERRIDES"

DEFAULTS: Dict[str, Any] = {
    "recursion": {
        "max_depth": 2,
        "branches": 3,
        "time_budget_s": 30,
        "token_budget": 2048,
        "novelty_floor": 0.15,
        "pertinence_tau": 0.35,
        "self_consistency_k": 3,
        "leash": True,
        "dry_run": True,
    },
    "watchdog": {
        "cpu_pct": 85,
        "wall_timeout_s": 60,
        "kill_on_tool_loop": True,
    },
    "memory": {
        "topk_retrieval": 8,
        "vector_store": "faiss",
        "embedder": "sentence-transformers/all-MiniLM-L6-v2",
        "persist_path": "./artifacts/memory",
        "sqlite_path": "./artifacts/memory/fts.db",
    },
    "self_model": {
        "persona": "Genie Lamp (Lumaeth)",
        "standing_goals": [
            "assist user",
            "minimize risk",
            "explain reasoning",
        ],
        "constraints": ["respect leash", "stay offline unless asked"],
        "confidence_decay_days": 21,
    },
    "tts": {
        "voice": "default",
        "fallback_voice": "english",
        "rate": 185,
        "volume": 0.9,
        "async_playback": False,
        "persist_audio": False,
        "cache_dir": "./artifacts/audio_cache",
        "playback_backend": "pydub",
        "driver": "espeak",
        "preload_samples": [
            "Genie Lamp boot sequence complete.",
            "Your Linux synthesizer is ready.",
        ],
    },
    "vision": {"ocr_lang": "eng"},
    "actions": {
        "allowlist_windows": [
            "Notepad",
            "Word",
            "Chrome",
            "Firefox",
            "Explorer",
        ],
        "email": {
            "smtp_host": "smtp.example.com",
            "smtp_port": 465,
            "smtp_user": "",
            "smtp_pass_env": "GENIE_SMTP_PASS",
        },
    },
    "speech": {
        "wakeword": "lumaeth",
        "stt_model": "medium",
        "vad": True,
    },
    "scheduler": {
        "dreams_cron": "0 3 * * *",
        "enable": True,
    },
    "retrieval": {
        "hybrid": True,
        "use_fts5": True,
        "min_confidence": 0.35,
        "faiss": {"dim": 768, "index": "Flat"},
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "cache_dir": "./artifacts/models/sentence-transformers",
    },
    "router": {
        "default_model": "local",
        "routes": {"chat": "local", "analysis": "analyzer", "plan": "planner"},
    },
    "preferences": {
        "profile_path": "./artifacts/preferences.json",
        "learning_rate": 0.2,
    },
    "skills": {
        "library_path": "./artifacts/skills",
        "autocompile": True,
    },
    "adapters": {
        "enabled": True,
        "base_model": "local-transformer",
        "peft_dir": "./artifacts/adapters",
    },
    "reasoners": {
        "enable_logic": True,
        "enable_causal": True,
    },
    "rehearsal": {"desktop_twin": {"enable": True, "workspace": "./artifacts/desktop_twin"}},
    "drives": {
        "enable_curiosity": True,
        "enable_resilience": True,
        "baseline_motivation": 0.6,
    },
    "shield": {
        "enable": True,
        "forbidden_terms": ["self-harm", "exploit", "malware"],
    },
    "observability": {"enable": True, "host": "127.0.0.1", "port": 8042},
    "models": {
        "transformer": "gpt2",
        "transformers_cache": "./artifacts/models/transformers",
    },
    "logging": {
        "level": "INFO",
        "json_path": "./artifacts/logs/genie_lamp.log.jsonl",
    },
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_mapping(text: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        if yaml is not None:
            parsed_yaml = yaml.safe_load(text)
            if isinstance(parsed_yaml, dict):
                return parsed_yaml
    return {}


def _load_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return _load_mapping(path.read_text(encoding="utf-8"))


def load_config() -> Dict[str, Any]:
    cfg = deepcopy(DEFAULTS)

    if DEFAULT_CONFIG_PATH.exists():
        cfg = _deep_update(cfg, _load_file(DEFAULT_CONFIG_PATH))

    env_path = os.environ.get("GENIE_LAMP_CONFIG")
    if env_path:
        cfg = _deep_update(cfg, _load_file(Path(env_path)))

    env_override = os.environ.get(ENV_OVERRIDE_KEY)
    if env_override:
        cfg = _deep_update(cfg, _load_mapping(env_override))

    return cfg


__all__ = ["load_config", "DEFAULTS", "DEFAULT_CONFIG_PATH", "PROJECT_ROOT"]

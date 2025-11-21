from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

from genie_lamp.core.config import DEFAULT_CONFIG_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genie Lamp service runner")
    parser.add_argument("command", nargs="?", default="serve", choices=["serve"], help="Action to execute")
    parser.add_argument("--host", default=os.environ.get("GENIE_LAMP_HOST", "127.0.0.1"), help="Host interface to bind")
    parser.add_argument("--port", type=int, default=int(os.environ.get("GENIE_LAMP_PORT", 7860)), help="Port to expose")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload (dev)")
    parser.add_argument("--config", type=Path, help="Path to configuration file (JSON or YAML)")
    parser.add_argument("--log-level", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], help="Override Loguru log level")
    parser.add_argument("--use-gpu", action="store_true", help="Attempt to enable GPU execution if available")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    module_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PYTHONPATH", str(module_root))
    os.environ.setdefault("GENIE_LAMP_HOME", str(module_root.parent.parent))

    if args.config:
        os.environ["GENIE_LAMP_CONFIG"] = str(args.config.resolve())
    else:
        os.environ.setdefault("GENIE_LAMP_CONFIG", str(DEFAULT_CONFIG_PATH))

    if args.log_level:
        os.environ["GENIE_LAMP_LOG_LEVEL"] = args.log_level

    os.environ["GENIE_LAMP_USE_GPU"] = "1" if args.use_gpu else os.environ.get("GENIE_LAMP_USE_GPU", "0")

    invoked_as_module = Path(sys.argv[0]).name == "__main__.py"
    reload_enabled = bool(args.reload or invoked_as_module)

    if args.command != "serve":  # pragma: no cover - defensive
        parser.error(f"Unsupported command: {args.command}")

    uvicorn.run(
        "genie_lamp.main:app",
        host=args.host,
        port=args.port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()

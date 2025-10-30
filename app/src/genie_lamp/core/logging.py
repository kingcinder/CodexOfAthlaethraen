from __future__ import annotations

import os
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from genie_lamp.core.config import PROJECT_ROOT

_request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_LOG_INITIALIZED = False


def _patch_record(record: Dict[str, Any]) -> None:
    if record["extra"].get("request_id") is None:
        record["extra"]["request_id"] = _request_id_ctx.get()


def configure_logging(cfg: Dict[str, Any]) -> None:
    global _LOG_INITIALIZED
    if _LOG_INITIALIZED:
        return

    log_level = os.environ.get("GENIE_LAMP_LOG_LEVEL", cfg.get("logging", {}).get("level", "INFO"))

    log_dir = PROJECT_ROOT / "artifacts" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = f"app-{datetime.utcnow():%Y%m%d}.jsonl"
    log_path = log_dir / log_filename

    logger.remove()
    logger.configure(patcher=_patch_record, extra={"request_id": None})

    def _console_format(record: Dict[str, Any]) -> str:
        rid = record["extra"].get("request_id")
        request_fragment = f" [req:{rid}]" if rid else ""
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "| <level>{level: <8}</level>"
            f"{request_fragment} | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
            "<level>{message}</level>"
        )

    logger.add(
        sys.stderr,
        level=log_level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format=_console_format,
    )

    logger.add(
        log_path,
        level=log_level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        serialize=True,
        rotation="00:00",  # rotate daily
        retention="14 days",
    )

    logger.bind(log_path=str(log_path)).info("Logging configured", path=str(log_path))
    _LOG_INITIALIZED = True


def get_request_id() -> Optional[str]:
    return _request_id_ctx.get()


class RequestContextLogMiddleware(BaseHTTPMiddleware):
    """Attach a request identifier and log lifecycle events."""

    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(self.header_name)
        request_id = incoming or str(uuid4())
        token = _request_id_ctx.set(request_id)
        bound_logger = logger.bind(request_id=request_id, path=str(request.url.path))
        bound_logger.info("Request started", method=request.method)
        try:
            response = await call_next(request)
        finally:
            _request_id_ctx.reset(token)
        response.headers[self.header_name] = request_id
        bound_logger.info("Request completed", status=response.status_code)
        return response


__all__ = ["configure_logging", "get_request_id", "RequestContextLogMiddleware"]

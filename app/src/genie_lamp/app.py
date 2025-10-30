from __future__ import annotations

from fastapi import FastAPI
from loguru import logger

from genie_lamp.api.routes import register_routes
from genie_lamp.core.agent import GenieAgent
from genie_lamp.core.config import load_config
from genie_lamp.core.logging import configure_logging, RequestContextLogMiddleware


def create_app() -> FastAPI:
    cfg = load_config()
    configure_logging(cfg)
    logger.bind(component="bootstrap").info(
        "Booting Genie Lamp", retrieval=cfg.get("retrieval"), models=cfg.get("models")
    )

    app = FastAPI(title="Genie Lamp", version="1.0")
    app.add_middleware(RequestContextLogMiddleware)

    agent = GenieAgent(cfg)
    app.state.agent = agent
    app.state.config = cfg

    register_routes(app)
    return app


__all__ = ["create_app"]

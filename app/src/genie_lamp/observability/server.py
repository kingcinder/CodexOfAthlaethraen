import threading
from typing import Any, Dict

from fastapi import FastAPI
import uvicorn


class ObservabilityServer:
    """Exposes a lightweight HTTP API with agent diagnostics."""

    def __init__(self, agent, cfg: dict):
        obs_cfg = cfg.get("observability", {})
        self.agent = agent
        self.host = obs_cfg.get("host", "127.0.0.1")
        self.port = obs_cfg.get("port", 8042)
        self.enabled = obs_cfg.get("enable", False)
        self._app = FastAPI(title="Genie Lamp Observability", docs_url=None, redoc_url=None)
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._configure_routes()

    def _configure_routes(self) -> None:
        @self._app.get("/health")
        def health() -> Dict[str, Any]:
            return {"status": "ok", "lantern": bool(self.agent.lantern)}

        @self._app.get("/memory")
        def memory_snapshot(limit: int = 5) -> Dict[str, Any]:
            return {"recent": self.agent.mem.recent(limit)}

        @self._app.get("/drives")
        def drives() -> Dict[str, Any]:
            return self.agent.drives.snapshot()

        @self._app.get("/preferences")
        def preferences() -> Dict[str, Any]:
            return self.agent.preferences.snapshot()

        @self._app.get("/router")
        def router() -> Dict[str, Any]:
            return self.agent.router.describe()

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        config = uvicorn.Config(self._app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def started(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

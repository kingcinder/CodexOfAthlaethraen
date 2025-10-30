from typing import Dict, Optional


class ModelRouter:
    """Simple policy-based router that selects a model for a given intent."""

    def __init__(self, cfg: dict):
        router_cfg = cfg.get("router", {})
        self.default_model = router_cfg.get("default_model", "local")
        self.routes: Dict[str, str] = router_cfg.get("routes", {})
        self.fallback = self.default_model

    def select_route(self, intent: str, metadata: Optional[dict] = None) -> str:
        metadata = metadata or {}
        normalized = (intent or "").lower()
        if normalized in self.routes:
            return self.routes[normalized]
        topic = metadata.get("topic")
        if topic and topic in self.routes:
            return self.routes[topic]
        return self.default_model

    def describe(self) -> Dict[str, str]:
        return {"default": self.default_model, "routes": self.routes}

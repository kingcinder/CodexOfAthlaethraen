from typing import Dict, List


class PromptShield:
    """Performs lightweight filtering and redaction on user prompts."""

    def __init__(self, cfg: dict):
        shield_cfg = cfg.get("shield", {})
        self.enabled = shield_cfg.get("enable", False)
        self.forbidden: List[str] = [t.lower() for t in shield_cfg.get("forbidden_terms", [])]

    def filter(self, text: str) -> Dict[str, object]:
        cleaned = text
        blocked = False
        if self.enabled:
            lowered = text.lower()
            for token in self.forbidden:
                if token in lowered:
                    cleaned = cleaned.replace(token, "[redacted]")
                    blocked = True
        return {"text": cleaned, "blocked": blocked}

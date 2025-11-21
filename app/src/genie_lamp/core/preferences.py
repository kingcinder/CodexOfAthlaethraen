import json
from pathlib import Path
from typing import Dict


class PreferenceModel:
    """Stores lightweight preference statistics for the assistant."""

    def __init__(self, cfg: dict):
        pref_cfg = cfg.get("preferences", {})
        self.path = Path(pref_cfg.get("profile_path", "./data/preferences.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.learning_rate = pref_cfg.get("learning_rate", 0.1)
        self.state: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.state = {}

    def _persist(self) -> None:
        self.path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")

    def observe(self, category: str, score: float) -> None:
        current = self.state.get(category, 0.5)
        updated = current + self.learning_rate * (score - current)
        self.state[category] = max(0.0, min(1.0, updated))
        self._persist()

    def update_from_interaction(self, user_text: str, assistant_text: str) -> None:
        sentiment = 1.0 if "thank" in assistant_text.lower() else 0.6
        urgency = 0.8 if any(token in user_text.lower() for token in ["urgent", "asap", "important"]) else 0.4
        self.observe("sentiment", sentiment)
        self.observe("urgency", urgency)

    def snapshot(self) -> Dict[str, float]:
        return dict(self.state)

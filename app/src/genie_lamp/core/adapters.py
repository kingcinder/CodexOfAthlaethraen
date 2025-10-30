from pathlib import Path
from typing import Dict, List


class AdapterManager:
    """Tracks PEFT-style adapter weights and activation state."""

    def __init__(self, cfg: dict):
        adapters_cfg = cfg.get("adapters", {})
        self.enabled = adapters_cfg.get("enabled", False)
        self.base_model = adapters_cfg.get("base_model", "")
        self.peft_dir = Path(adapters_cfg.get("peft_dir", "./data/adapters"))
        self.peft_dir.mkdir(parents=True, exist_ok=True)
        self.active_adapter: str | None = None

    def available(self) -> List[str]:
        return sorted({p.stem for p in self.peft_dir.glob("*.bin")})

    def activate(self, name: str) -> bool:
        if name in self.available():
            self.active_adapter = name
            return True
        return False

    def deactivate(self) -> None:
        self.active_adapter = None

    def describe(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "base_model": self.base_model,
            "active": self.active_adapter,
            "available": self.available(),
        }

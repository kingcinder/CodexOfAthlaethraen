from typing import Dict


class DriveEngine:
    """Tracks intrinsic motivation signals for the agent."""

    def __init__(self, cfg: dict):
        drive_cfg = cfg.get("drives", {})
        self.enable_curiosity = drive_cfg.get("enable_curiosity", False)
        self.enable_resilience = drive_cfg.get("enable_resilience", False)
        self.baseline = drive_cfg.get("baseline_motivation", 0.5)
        self.state: Dict[str, float] = {
            "curiosity": self.baseline if self.enable_curiosity else 0.0,
            "resilience": self.baseline if self.enable_resilience else 0.0,
        }

    def apply_feedback(self, result: Dict[str, object]) -> None:
        text = str(result.get("text", ""))
        novelty = 0.7 if "new" in text.lower() else 0.4
        success = 0.9 if result.get("text") else 0.5
        if self.enable_curiosity:
            self.state["curiosity"] = self._blend(self.state.get("curiosity", self.baseline), novelty)
        if self.enable_resilience:
            self.state["resilience"] = self._blend(self.state.get("resilience", self.baseline), success)

    def _blend(self, current: float, target: float) -> float:
        return max(0.0, min(1.0, current * 0.7 + target * 0.3))

    def snapshot(self) -> Dict[str, float]:
        return dict(self.state)

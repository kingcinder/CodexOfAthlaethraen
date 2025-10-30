import json
from pathlib import Path
from typing import Dict, List


class DesktopTwin:
    """Captures rehearsal logs for desktop task simulations."""

    def __init__(self, cfg: dict, memory, reasoners):
        rehearsal_cfg = cfg.get("rehearsal", {}).get("desktop_twin", {})
        self.enabled = rehearsal_cfg.get("enable", False)
        self.workspace = Path(rehearsal_cfg.get("workspace", "./data/desktop_twin"))
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.memory = memory
        self.reasoners = reasoners

    def record_session(self, prompt: str, result: Dict[str, object], recall: List[object]) -> None:
        if not self.enabled:
            return
        session = {
            "prompt": prompt,
            "result": result,
            "recall": recall,
            "reasoning": self.reasoners.evaluate(prompt, {"facts": [r for r in recall]}) if self.reasoners else {},
        }
        path = self.workspace / "rehearsals.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(session, ensure_ascii=False) + "\n")

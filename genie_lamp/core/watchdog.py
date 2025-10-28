"""Runtime guardrails for tool execution."""

from __future__ import annotations

import time
from typing import List

import psutil


class Watchdog:
    """Monitors resource usage and simple tool repetition loops."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.t0 = time.time()
        self.tool_seq: List[str] = []

    def reset(self) -> None:
        self.t0 = time.time()
        self.tool_seq.clear()

    def record_tool(self, name: str) -> None:
        self.tool_seq.append(name)

    def ok(self) -> bool:
        watchdog_cfg = self.cfg.get("watchdog", {})
        cpu_limit = watchdog_cfg.get("cpu_pct", 100)
        wall_timeout = watchdog_cfg.get("wall_timeout_s", float("inf"))
        allow_loop_break = watchdog_cfg.get("kill_on_tool_loop", False)

        if psutil.cpu_percent(interval=0.1) > cpu_limit:
            return False
        if time.time() - self.t0 > wall_timeout:
            return False
        if allow_loop_break and self._recent_loop_detected():
            return False
        return True

    def _recent_loop_detected(self) -> bool:
        if len(self.tool_seq) < 3:
            return False
        window = self.tool_seq[-3:]
        return len(set(window)) == 1

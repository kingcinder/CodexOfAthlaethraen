from typing import Any, Dict, List, Tuple

from core.hybrid_retrieval import HybridRetriever
from core.vector_store import VectorStore

class Memory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.vs = VectorStore(cfg)
        retrieval_cfg = cfg.get("retrieval", {})
        self.hybrid = HybridRetriever(cfg, self.vs) if retrieval_cfg.get("hybrid") else None
        self.timeline: List[Dict[str, Any]] = []

    def remember(self, item: dict):
        text, meta = self._prepare_item(item)
        self.vs.upsert([(text, meta)])
        if self.hybrid:
            self.hybrid.upsert([(text, meta)])
        self.timeline.append(item)

    def recall(self, query: str, top_k: int = 8):
        if self.hybrid and self.cfg.get("retrieval", {}).get("hybrid"):
            min_conf = self.cfg.get("retrieval", {}).get("min_confidence", 0.0)
            return self.hybrid.search(query, k=top_k, min_confidence=min_conf)
        return self.vs.search(query, k=top_k)

    def write_reflection(self, refl_json: str):
        payload = {"type": "reflection", "content": refl_json}
        text, meta = self._prepare_item(payload)
        self.vs.upsert([(text, meta)])
        if self.hybrid:
            self.hybrid.upsert([(text, meta)])
        self.timeline.append(payload)

    def recent(self, limit: int = 5):
        return self.timeline[-limit:]

    def _prepare_item(self, item: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        text = (
            str(item.get("assistant"))
            if item.get("assistant")
            else str(item.get("user") or item)
        )
        meta = {"kind": item.get("type", "note")}
        for key, value in item.items():
            if key in {"assistant", "user"}:
                continue
            if isinstance(value, (str, int, float)):
                meta[key] = value
        return text, meta

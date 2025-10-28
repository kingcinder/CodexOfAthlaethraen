import json
import os
import sqlite3
from typing import Iterable, List, Tuple, Dict, Any


class HybridRetriever:
    """Combines vector search with SQLite FTS5 for hybrid recall."""

    def __init__(self, cfg: dict, vector_store):
        self.cfg = cfg
        self.vector_store = vector_store
        memory_cfg = cfg.get("memory", {})
        self.sqlite_path = memory_cfg.get("sqlite_path", "./data/fts.db")
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        self.conn = sqlite3.connect(self.sqlite_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS hybrid_docs USING fts5(doc, meta)"
        )
        self.conn.commit()

    def upsert(self, items: Iterable[Tuple[str, Dict[str, Any]]]) -> None:
        cur = self.conn.cursor()
        for text, meta in items:
            cur.execute(
                "INSERT INTO hybrid_docs(doc, meta) VALUES(?, ?)",
                (text, json.dumps(meta, ensure_ascii=False)),
            )
        self.conn.commit()

    def search(self, query: str, k: int = 8, min_confidence: float = 0.0):
        results: List[Tuple[str, Dict[str, Any], float]] = []
        if not query:
            return []
        cur = self.conn.cursor()
        try:
            cur.execute(
                "SELECT doc, meta, bm25(hybrid_docs) as score FROM hybrid_docs WHERE hybrid_docs MATCH ? ORDER BY score LIMIT ?",
                (query, k),
            )
            for doc, meta, score in cur.fetchall():
                confidence = max(1.0 - (score or 0.0) / 10.0, 0.0)
                if confidence >= min_confidence:
                    results.append((doc, json.loads(meta), confidence))
        except sqlite3.OperationalError:
            # FTS query failed (likely due to unsupported characters); fall back silently
            pass

        vector_hits = self.vector_store.search(query, k=k)
        merged: Dict[str, Tuple[Dict[str, Any], float]] = {}
        for doc, meta in vector_hits:
            merged[doc] = (meta, merged.get(doc, ({}, 0.0))[1])
        for doc, meta, confidence in results:
            existing = merged.get(doc)
            if existing:
                merged[doc] = (meta, max(confidence, existing[1]))
            else:
                merged[doc] = (meta, confidence)

        ordered = sorted(merged.items(), key=lambda item: item[1][1], reverse=True)
        return [
            {"text": doc, "metadata": meta, "confidence": conf}
            for doc, (meta, conf) in ordered[:k]
        ]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

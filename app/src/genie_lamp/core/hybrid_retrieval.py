import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


class HybridRetriever:
    """Combines vector search with SQLite FTS5 for hybrid recall."""

    def __init__(self, cfg: dict, vector_store):
        self.cfg = cfg
        self.vector_store = vector_store
        memory_cfg = cfg.get("memory", {})
        sqlite_path_cfg = memory_cfg.get("sqlite_path", "./artifacts/memory/fts.db")
        sqlite_path = Path(sqlite_path_cfg)
        if not sqlite_path.is_absolute():
            project_root = Path(__file__).resolve().parents[3]
            sqlite_path = (project_root / sqlite_path).resolve()
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = str(sqlite_path)
        self.conn = sqlite3.connect(self.sqlite_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.fts_available = False
        self.fts_error: str | None = None
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS hybrid_meta (doc TEXT PRIMARY KEY, meta TEXT NOT NULL)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_hybrid_meta_doc ON hybrid_meta(doc)"
            )
        try:
            with self.conn:
                self.conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS hybrid_docs USING fts5(doc, meta)"
                )
            self.fts_available = True
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).lower():
                self.fts_available = False
                self.fts_error = str(exc)
            else:
                raise
        with self.conn:
            self.conn.execute("ANALYZE hybrid_meta")
            if self.fts_available:
                self.conn.execute("ANALYZE hybrid_docs")

    @staticmethod
    def _normalize_vector_score(score: float) -> float:
        value = (float(score) + 1.0) / 2.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _normalize_bm25(score: float) -> float:
        clamped = max(float(score), 0.0)
        return 1.0 / (1.0 + clamped)

    def upsert(self, items: Iterable[Tuple[str, Dict[str, Any]]]) -> None:
        payloads = [(text, json.dumps(meta, ensure_ascii=False)) for text, meta in items]
        if not payloads:
            return
        with self.conn:
            for text, meta_json in payloads:
                self.conn.execute(
                    "INSERT OR REPLACE INTO hybrid_meta(doc, meta) VALUES (?, ?)",
                    (text, meta_json),
                )
                if self.fts_available:
                    self.conn.execute("DELETE FROM hybrid_docs WHERE doc = ?", (text,))
                    self.conn.execute(
                        "INSERT INTO hybrid_docs(doc, meta) VALUES (?, ?)",
                        (text, meta_json),
                    )

    def search(self, query: str, k: int = 8, min_confidence: float = 0.0):
        """Return merged FTS and vector results with scores normalised to [0, 1]."""

        if not query:
            return []

        results: List[Tuple[str, Dict[str, Any], float]] = []
        if self.fts_available:
            try:
                with self.conn as conn:
                    cursor = conn.execute(
                        "SELECT doc, meta, bm25(hybrid_docs) AS score FROM hybrid_docs WHERE hybrid_docs MATCH ? ORDER BY score LIMIT ?",
                        (query, k),
                    )
                    rows = cursor.fetchall()
            except sqlite3.OperationalError:
                rows = []
            for doc, meta_json, raw_score in rows:
                confidence = self._normalize_bm25(raw_score)
                if confidence >= min_confidence:
                    results.append((doc, json.loads(meta_json), confidence))

        vector_hits = self.vector_store.search(query, k=k)
        merged: Dict[str, Tuple[Dict[str, Any], float]] = {}

        # Normalise cosine similarities to [0, 1] and take the maximum score
        # when the same document appears in both retrieval modalities.
        for item in vector_hits:
            if isinstance(item, dict):
                doc = item.get("text", "")
                meta = item.get("meta", {})
                base_score = item.get("score", 0.0) or 0.0
            else:
                doc, meta = item  # type: ignore[misc]
                base_score = 0.0
            if not doc:
                continue
            vector_score = self._normalize_vector_score(base_score)
            merged[doc] = (meta, max(vector_score, merged.get(doc, ({}, 0.0))[1]))

        for doc, meta, confidence in results:
            existing = merged.get(doc)
            if existing:
                merged[doc] = (meta, max(confidence, existing[1]))
            else:
                merged[doc] = (meta, confidence)

        ordered = sorted(merged.items(), key=lambda item: item[1][1], reverse=True)
        return [
            {"text": doc, "meta": meta, "score": float(conf)}
            for doc, (meta, conf) in ordered[:k]
        ]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

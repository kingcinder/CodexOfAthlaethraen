from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    faiss = None  # type: ignore

from genie_lamp.core.models import load_sentence_encoder


class VectorStore:
    def __init__(self, cfg: Dict[str, Any]):
        self.encoder = load_sentence_encoder(cfg)
        self.dim = int(self.encoder.get_sentence_embedding_dimension())
        self._texts: List[str] = []
        self._meta: List[Dict[str, Any]] = []
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = None
            self._mem = np.zeros((0, self.dim), dtype="float32")

    def _embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def upsert(self, items: List[Tuple[str, Dict[str, Any]]]) -> None:
        if not items:
            return
        texts = [text for text, _ in items]
        metadata = [meta for _, meta in items]
        vectors = self._embed(texts)
        self._texts.extend(texts)
        self._meta.extend(metadata)
        if self.index is not None:
            self.index.add(vectors)
        else:
            self._mem = np.vstack([self._mem, vectors]) if self._mem.size else vectors

    def search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        if not self._texts:
            return []
        query_vec = self._embed([query])
        if self.index is not None:
            scores, indices = self.index.search(query_vec, k)
            idx_list = indices[0].tolist()
            score_list = scores[0].tolist()
        else:
            sims = self._mem @ query_vec[0]
            order = np.argsort(-sims)[:k]
            idx_list = order.tolist()
            score_list = sims[order].tolist()
        results: List[Dict[str, Any]] = []
        for idx, score in zip(idx_list, score_list):
            if idx < 0 or idx >= len(self._texts):
                continue
            results.append(
                {"text": self._texts[idx], "meta": self._meta[idx], "score": float(score)}
            )
        return results


__all__ = ["VectorStore"]

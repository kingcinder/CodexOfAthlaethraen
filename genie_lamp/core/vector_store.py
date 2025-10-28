from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg["memory"]["embedder"])
        self.backend = cfg["memory"]["vector_store"]
        if self.backend == "chroma":
            import chromadb
            self.client = chromadb.PersistentClient(path=cfg["memory"]["persist_path"])
            self.col = self.client.get_or_create_collection("genie_mem")
        else:
            import faiss, numpy as np
            self.faiss, self.np = faiss, np
            self.index, self.docs = None, []

    def upsert(self, items):
        texts = [t for t,_ in items]; metas = [m for _,m in items]
        embs = self.model.encode(texts).tolist()
        if self.backend == "chroma":
            ids = [f"id_{i}" for i,_ in enumerate(texts)]
            self.col.add(ids=ids, metadatas=metas, documents=texts, embeddings=embs)
        else:
            x = self.np.array(embs, dtype="float32")
            if self.index is None: self.index = self.faiss.IndexFlatIP(x.shape[1])
            self.index.add(x); self.docs.extend(list(zip(texts, metas)))

    def search(self, query: str, k: int = 8):
        q = self.model.encode([query]).tolist()[0]
        if self.backend == "chroma":
            res = self.col.query(query_embeddings=[q], n_results=k)
            return list(zip(res["documents"][0], res["metadatas"][0]))
        else:
            xq = self.np.array([q], dtype="float32")
            if self.index is None: return []
            D, I = self.index.search(xq, k)
            return [self.docs[i] for i in I[0] if 0 <= i < len(self.docs)]

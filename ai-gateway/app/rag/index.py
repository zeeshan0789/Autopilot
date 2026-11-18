"""A FAISS index over chunk embeddings. The embedder must return normalized vectors, so the inner
product is the cosine similarity."""

from dataclasses import dataclass

import faiss
import numpy as np

from app.rag.chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int  # 1 is the best


class ChunkIndex:
    def __init__(self, chunks: list[Chunk], embedder):
        if not chunks:
            raise ValueError("no chunks to index")
        self.chunks = list(chunks)
        self.embedder = embedder
        vectors = np.vstack([np.asarray(embedder.encode(c.embedding_text), dtype=np.float32) for c in self.chunks])
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(np.ascontiguousarray(vectors))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        vector = np.asarray(self.embedder.encode(query), dtype=np.float32).reshape(1, -1)
        scores, ids = self._index.search(vector, min(k, len(self.chunks)))
        return [
            Hit(self.chunks[int(i)], float(s), rank)
            for rank, (s, i) in enumerate(zip(scores[0], ids[0]), 1)
            if int(i) >= 0
        ]

    def chunks_of(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]
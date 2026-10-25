import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable

import faiss
import numpy as np


@dataclass(frozen=True)
class CacheScope:
    """Entries are only ever compared inside the same scope.

    A cached answer is safe to reuse only for the same caller (namespace), the
    same model, the same system prompt, the same temperature bucket and the same
    prompt version, so all of them are part of the partition key.
    """

    namespace: str
    model: str
    system_hash: str
    temperature_bucket: str
    prompt_version: str = "none"


@dataclass
class CacheEntry:
    prompt_text: str  # kept for the safety guard (numbers, years, names)
    response: dict
    served_model: str
    provider: str
    prompt_tokens: int  # tokens of the original call, used to measure savings
    completion_tokens: int
    estimated_cost_usd: Decimal | None
    created_at: float
    ttl_seconds: float
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def is_expired(self, now: float) -> bool:
        return now >= self.created_at + self.ttl_seconds


@dataclass
class CacheCandidate:
    """The closest entry found. Whether it is good enough is the caller's decision."""

    entry: CacheEntry
    similarity: float  # cosine similarity, 1.0 means the same direction


def _unit(vector) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("embedding must not be a zero vector")
    return arr / norm


class _Partition:
    """Entries for one scope, plus a FAISS index over their vectors."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.entries: list[CacheEntry] = []
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        # Inner product on unit vectors equals cosine similarity
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, vector: np.ndarray, entry: CacheEntry) -> None:
        row = vector.reshape(1, -1)
        self.entries.append(entry)
        self.vectors = np.vstack([self.vectors, row])
        self.index.add(row)

    def keep_only(self, mask: list[bool]) -> None:
        """Drop entries where mask is False, then rebuild the FAISS index."""
        self.entries = [e for e, keep in zip(self.entries, mask) if keep]
        self.vectors = np.ascontiguousarray(self.vectors[np.array(mask, dtype=bool)])
        self.index = faiss.IndexFlatIP(self.dimension)
        if self.entries:
            self.index.add(self.vectors)


class SemanticCache:
    def __init__(
        self,
        dimension: int = 384,
        max_entries_per_scope: int = 1000,
        clock: Callable[[], float] = time.time,
    ):
        self.dimension = dimension
        self.max_entries_per_scope = max_entries_per_scope
        self._clock = clock
        self._partitions: dict[CacheScope, _Partition] = {}

    def add(self, scope: CacheScope, embedding, entry: CacheEntry) -> None:
        vector = self._prepare(embedding)
        partition = self._partitions.get(scope)
        if partition is None:
            partition = _Partition(self.dimension)
            self._partitions[scope] = partition

        self._drop_expired(partition)
        partition.add(vector, entry)

        # Over the limit: drop the oldest entries first
        overflow = len(partition.entries) - self.max_entries_per_scope
        if overflow > 0:
            partition.keep_only([i >= overflow for i in range(len(partition.entries))])

    def lookup(self, scope: CacheScope, embedding) -> CacheCandidate | None:
        vector = self._prepare(embedding)
        partition = self._partitions.get(scope)
        if partition is None:
            return None

        self._drop_expired(partition)
        if not partition.entries:
            return None

        scores, ids = partition.index.search(vector.reshape(1, -1), 1)
        position = int(ids[0][0])
        if position < 0:
            return None
        return CacheCandidate(partition.entries[position], float(scores[0][0]))

    def lookup_many(
        self, scope: CacheScope, embedding, k: int = 5
    ) -> list[CacheCandidate]:
        """The k closest live entries, best first. Applying a threshold is the caller's job."""
        vector = self._prepare(embedding)
        partition = self._partitions.get(scope)
        if partition is None:
            return []

        self._drop_expired(partition)
        if not partition.entries:
            return []

        count = min(k, len(partition.entries))
        scores, ids = partition.index.search(vector.reshape(1, -1), count)
        return [
            CacheCandidate(partition.entries[int(i)], float(s))
            for s, i in zip(scores[0], ids[0])
            if int(i) >= 0
        ]

    def size(self) -> int:
        return sum(len(p.entries) for p in self._partitions.values())

    def _prepare(self, embedding) -> np.ndarray:
        vector = _unit(embedding)
        if vector.shape[0] != self.dimension:
            raise ValueError(
                f"embedding has {vector.shape[0]} dimensions, expected {self.dimension}"
            )
        return np.ascontiguousarray(vector)

    def _drop_expired(self, partition: _Partition) -> None:
        now = self._clock()
        mask = [not e.is_expired(now) for e in partition.entries]
        if not all(mask):
            partition.keep_only(mask)
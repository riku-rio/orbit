from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Protocol, Sequence

from orbit.memory.config import MemoryConfig
from orbit.memory.errors import MemoryModelError, MemoryValidationError
from orbit.memory.types import (
    MemoryAddResult,
    MemoryCandidate,
    MemoryHit,
    MemoryRecord,
)

MEMORY_SCHEMA_VERSION = 1
MEMORY_ID_NAMESPACE = uuid.UUID("168f9a22-1c87-4fcf-a276-d79cecf0f4d4")


class MemoryModelProvider(Protocol):
    def validate_memory(self, text: str) -> None: ...

    def embed_memory(self, text: str) -> list[float]: ...

    def embed_query(self, query: str) -> list[float]: ...

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]: ...


class MemoryStore(Protocol):
    def get(self, point_id: str) -> MemoryRecord | None: ...

    def upsert(self, record: MemoryRecord, vector: list[float]) -> None: ...

    def search(self, vector: list[float], *, limit: int) -> list[MemoryCandidate]: ...


def normalize_memory_text(text: str) -> str:
    return " ".join(text.split())


def memory_identity(text: str) -> tuple[str, str]:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    point_id = str(uuid.uuid5(MEMORY_ID_NAMESPACE, content_hash))
    return point_id, content_hash


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MemoryService:
    def __init__(
        self,
        config: MemoryConfig,
        models: MemoryModelProvider,
        store: MemoryStore,
    ) -> None:
        self.config = config
        self.models = models
        self.store = store

    def add(self, memory: str) -> MemoryAddResult:
        if not isinstance(memory, str):
            raise MemoryValidationError("Memory must be a string.")
        text = normalize_memory_text(memory)
        if not text:
            raise MemoryValidationError("Memory cannot be empty.")

        point_id, content_hash = memory_identity(text)
        existing = self.store.get(point_id)
        if existing is not None:
            return MemoryAddResult(record=existing, already_stored=True)

        self.models.validate_memory(text)
        vector = self.models.embed_memory(text)
        if len(vector) != self.config.vector_size:
            raise MemoryModelError(
                f"Embedding produced {len(vector)} dimensions; "
                f"expected {self.config.vector_size}."
            )

        record = MemoryRecord(
            id=point_id,
            text=text,
            stored_at=_utc_now_iso(),
            content_hash=content_hash,
            schema_version=MEMORY_SCHEMA_VERSION,
            embedding_model=self.config.embedding_model,
            source="agent",
        )
        self.store.upsert(record, vector)
        return MemoryAddResult(record=record, already_stored=False)

    def retrieve(self, query: str, *, limit: int | None = None) -> list[MemoryHit]:
        if not isinstance(query, str):
            raise MemoryValidationError("Memory query must be a string.")
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise MemoryValidationError("Memory query cannot be empty.")

        if limit is None:
            result_limit = self.config.default_result_limit
        else:
            if not isinstance(limit, int) or isinstance(limit, bool):
                raise MemoryValidationError("Memory result limit must be an integer.")
            result_limit = max(1, min(limit, self.config.max_result_limit))

        query_vector = self.models.embed_query(normalized_query)
        if len(query_vector) != self.config.vector_size:
            raise MemoryModelError(
                f"Query embedding produced {len(query_vector)} dimensions; "
                f"expected {self.config.vector_size}."
            )

        candidate_limit = max(
            self.config.retrieval_candidates,
            result_limit * 4,
        )
        candidates = self.store.search(query_vector, limit=candidate_limit)
        if not candidates:
            return []

        rerank_scores = self.models.rerank(
            normalized_query,
            [candidate.record.text for candidate in candidates],
        )
        if len(rerank_scores) != len(candidates):
            raise MemoryModelError(
                "Reranker returned a different number of scores than candidates."
            )

        ranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda item: item[1],
            reverse=True,
        )[:result_limit]

        return [
            MemoryHit(
                rank=rank,
                record=candidate.record,
                vector_score=candidate.vector_score,
                rerank_score=float(rerank_score),
            )
            for rank, (candidate, rerank_score) in enumerate(ranked, start=1)
        ]

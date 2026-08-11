from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    text: str
    stored_at: str
    content_hash: str
    schema_version: int
    embedding_model: str
    source: str = "agent"

    def payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "stored_at": self.stored_at,
            "content_hash": self.content_hash,
            "schema_version": self.schema_version,
            "embedding_model": self.embedding_model,
            "source": self.source,
        }

    @classmethod
    def from_payload(
        cls,
        point_id: object,
        payload: dict[str, Any] | None,
    ) -> "MemoryRecord | None":
        if not isinstance(payload, dict):
            return None

        text = payload.get("text")
        stored_at = payload.get("stored_at")
        content_hash = payload.get("content_hash")
        schema_version = payload.get("schema_version")
        embedding_model = payload.get("embedding_model")
        source = payload.get("source", "agent")

        if not isinstance(text, str) or not text:
            return None
        if not isinstance(stored_at, str) or not stored_at:
            return None
        if not isinstance(content_hash, str) or not content_hash:
            return None
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            return None
        if not isinstance(embedding_model, str) or not embedding_model:
            return None
        if not isinstance(source, str) or not source:
            source = "agent"

        return cls(
            id=str(point_id),
            text=text,
            stored_at=stored_at,
            content_hash=content_hash,
            schema_version=schema_version,
            embedding_model=embedding_model,
            source=source,
        )


@dataclass(frozen=True)
class MemoryCandidate:
    record: MemoryRecord
    vector_score: float


@dataclass(frozen=True)
class MemoryAddResult:
    record: MemoryRecord
    already_stored: bool


@dataclass(frozen=True)
class MemoryHit:
    rank: int
    record: MemoryRecord
    vector_score: float
    rerank_score: float

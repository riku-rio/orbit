from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION_NAME = "orbit_memory_v1"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_VECTOR_SIZE = 1024
DEFAULT_RETRIEVAL_CANDIDATES = 20
DEFAULT_RESULT_LIMIT = 5
DEFAULT_MAX_RESULT_LIMIT = 10
DEFAULT_MAX_MEMORY_TOKENS = 480
DEFAULT_MAX_QUERY_TOKENS = 480
DEFAULT_RERANK_BATCH_SIZE = 8
DEFAULT_QDRANT_TIMEOUT_SECONDS = 5.0


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


@dataclass(frozen=True)
class MemoryConfig:
    qdrant_url: str = DEFAULT_QDRANT_URL
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    vector_size: int = DEFAULT_VECTOR_SIZE
    retrieval_candidates: int = DEFAULT_RETRIEVAL_CANDIDATES
    default_result_limit: int = DEFAULT_RESULT_LIMIT
    max_result_limit: int = DEFAULT_MAX_RESULT_LIMIT
    max_memory_tokens: int = DEFAULT_MAX_MEMORY_TOKENS
    max_query_tokens: int = DEFAULT_MAX_QUERY_TOKENS
    rerank_batch_size: int = DEFAULT_RERANK_BATCH_SIZE
    qdrant_timeout_seconds: float = DEFAULT_QDRANT_TIMEOUT_SECONDS
    device: str | None = None

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        device = os.getenv("ORBIT_MEMORY_DEVICE")
        if device is not None:
            device = device.strip() or None
        if device == "auto":
            device = None

        max_result_limit = _env_int(
            "ORBIT_MEMORY_MAX_RESULT_LIMIT", DEFAULT_MAX_RESULT_LIMIT
        )
        default_result_limit = min(
            _env_int("ORBIT_MEMORY_DEFAULT_RESULT_LIMIT", DEFAULT_RESULT_LIMIT),
            max_result_limit,
        )

        return cls(
            qdrant_url=os.getenv("ORBIT_QDRANT_URL", DEFAULT_QDRANT_URL).strip()
            or DEFAULT_QDRANT_URL,
            collection_name=os.getenv(
                "ORBIT_MEMORY_COLLECTION", DEFAULT_COLLECTION_NAME
            ).strip()
            or DEFAULT_COLLECTION_NAME,
            embedding_model=os.getenv(
                "ORBIT_MEMORY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            ).strip()
            or DEFAULT_EMBEDDING_MODEL,
            reranker_model=os.getenv(
                "ORBIT_MEMORY_RERANKER_MODEL", DEFAULT_RERANKER_MODEL
            ).strip()
            or DEFAULT_RERANKER_MODEL,
            vector_size=_env_int("ORBIT_MEMORY_VECTOR_SIZE", DEFAULT_VECTOR_SIZE),
            retrieval_candidates=_env_int(
                "ORBIT_MEMORY_RETRIEVAL_CANDIDATES",
                DEFAULT_RETRIEVAL_CANDIDATES,
            ),
            default_result_limit=default_result_limit,
            max_result_limit=max_result_limit,
            max_memory_tokens=_env_int(
                "ORBIT_MEMORY_MAX_MEMORY_TOKENS", DEFAULT_MAX_MEMORY_TOKENS
            ),
            max_query_tokens=_env_int(
                "ORBIT_MEMORY_MAX_QUERY_TOKENS", DEFAULT_MAX_QUERY_TOKENS
            ),
            rerank_batch_size=_env_int(
                "ORBIT_MEMORY_RERANK_BATCH_SIZE", DEFAULT_RERANK_BATCH_SIZE
            ),
            qdrant_timeout_seconds=_env_float(
                "ORBIT_QDRANT_TIMEOUT_SECONDS",
                DEFAULT_QDRANT_TIMEOUT_SECONDS,
                minimum=0.1,
            ),
            device=device,
        )

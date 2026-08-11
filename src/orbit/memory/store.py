from __future__ import annotations

from threading import Lock
from typing import Any

from orbit.memory.config import MemoryConfig
from orbit.memory.errors import MemoryBackendError, MemorySchemaError
from orbit.memory.types import MemoryCandidate, MemoryRecord


def _load_qdrant_dependencies() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except Exception as exc:
        raise MemoryBackendError(
            "qdrant-client is not installed or could not be imported."
        ) from exc
    return QdrantClient, models


def _distance_name(distance: object) -> str:
    value = getattr(distance, "value", distance)
    return str(value).strip().lower()


class QdrantMemoryStore:
    def __init__(
        self,
        config: MemoryConfig,
        *,
        client: Any | None = None,
        models_module: Any | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._models_module = models_module
        self._ready = False
        self._ready_lock = Lock()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        QdrantClient, models = _load_qdrant_dependencies()
        try:
            self._client = QdrantClient(
                url=self.config.qdrant_url,
                timeout=self.config.qdrant_timeout_seconds,
            )
        except Exception as exc:
            raise MemoryBackendError(
                f"Could not initialize Qdrant client for {self.config.qdrant_url}."
            ) from exc
        self._models_module = models
        return self._client

    def _get_models(self) -> Any:
        if self._models_module is not None:
            return self._models_module
        _, models = _load_qdrant_dependencies()
        self._models_module = models
        return models

    def _validate_collection(self, info: Any) -> None:
        try:
            vectors = info.config.params.vectors
        except Exception as exc:
            raise MemorySchemaError(
                f"Could not inspect collection '{self.config.collection_name}' schema."
            ) from exc

        if isinstance(vectors, dict):
            raise MemorySchemaError(
                f"Collection '{self.config.collection_name}' uses named vectors; "
                "Orbit memory requires one unnamed dense vector."
            )

        size = getattr(vectors, "size", None)
        distance = _distance_name(getattr(vectors, "distance", ""))
        if size != self.config.vector_size or distance != "cosine":
            raise MemorySchemaError(
                f"Collection '{self.config.collection_name}' schema mismatch: "
                f"expected vector size {self.config.vector_size} with cosine distance; "
                f"found size {size!r} with distance {distance or 'unknown'!r}."
            )

    def ensure_collection(self) -> None:
        if self._ready:
            return

        with self._ready_lock:
            if self._ready:
                return

            client = self._get_client()
            models = self._get_models()
            try:
                exists = client.collection_exists(self.config.collection_name)
                if not exists:
                    client.create_collection(
                        collection_name=self.config.collection_name,
                        vectors_config=models.VectorParams(
                            size=self.config.vector_size,
                            distance=models.Distance.COSINE,
                        ),
                    )
                info = client.get_collection(self.config.collection_name)
            except MemorySchemaError:
                raise
            except Exception as exc:
                raise MemoryBackendError(
                    "Memory backend unavailable at "
                    f"{self.config.qdrant_url}: {exc}"
                ) from exc

            self._validate_collection(info)
            self._ready = True

    def get(self, point_id: str) -> MemoryRecord | None:
        self.ensure_collection()
        client = self._get_client()
        try:
            points = client.retrieve(
                collection_name=self.config.collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise MemoryBackendError(f"Could not read memory from Qdrant: {exc}") from exc

        if not points:
            return None
        point = points[0]
        return MemoryRecord.from_payload(getattr(point, "id", point_id), point.payload)

    def upsert(self, record: MemoryRecord, vector: list[float]) -> None:
        self.ensure_collection()
        client = self._get_client()
        models = self._get_models()
        try:
            client.upsert(
                collection_name=self.config.collection_name,
                points=[
                    models.PointStruct(
                        id=record.id,
                        vector=vector,
                        payload=record.payload(),
                    )
                ],
                wait=True,
            )
        except Exception as exc:
            raise MemoryBackendError(f"Could not write memory to Qdrant: {exc}") from exc

    def search(self, vector: list[float], *, limit: int) -> list[MemoryCandidate]:
        self.ensure_collection()
        client = self._get_client()
        try:
            response = client.query_points(
                collection_name=self.config.collection_name,
                query=vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise MemoryBackendError(f"Could not retrieve memories from Qdrant: {exc}") from exc

        candidates: list[MemoryCandidate] = []
        for point in getattr(response, "points", []):
            record = MemoryRecord.from_payload(
                getattr(point, "id", ""),
                getattr(point, "payload", None),
            )
            if record is None:
                continue
            score = getattr(point, "score", 0.0)
            try:
                vector_score = float(score)
            except (TypeError, ValueError):
                vector_score = 0.0
            candidates.append(
                MemoryCandidate(record=record, vector_score=vector_score)
            )
        return candidates

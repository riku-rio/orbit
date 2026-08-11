from __future__ import annotations

from threading import Lock

from orbit.memory.config import MemoryConfig
from orbit.memory.models import MemoryModels
from orbit.memory.service import MemoryService
from orbit.memory.store import QdrantMemoryStore

_service: MemoryService | None = None
_service_lock = Lock()


def get_memory_service() -> MemoryService:
    global _service
    if _service is not None:
        return _service

    with _service_lock:
        if _service is None:
            config = MemoryConfig.from_env()
            _service = MemoryService(
                config=config,
                models=MemoryModels(config),
                store=QdrantMemoryStore(config),
            )
        return _service

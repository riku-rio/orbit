from __future__ import annotations


class OrbitMemoryError(RuntimeError):
    """Base error for Orbit's persistent memory subsystem."""


class MemoryValidationError(OrbitMemoryError):
    """Raised when a memory tool input is invalid."""


class MemoryBackendError(OrbitMemoryError):
    """Raised when Qdrant cannot be reached or queried."""


class MemorySchemaError(OrbitMemoryError):
    """Raised when an existing Qdrant collection has an incompatible schema."""


class MemoryModelError(OrbitMemoryError):
    """Raised when a local embedding or reranker model cannot be loaded or used."""

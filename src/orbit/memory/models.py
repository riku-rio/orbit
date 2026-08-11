from __future__ import annotations

import os
from collections.abc import Mapping
from threading import Lock
from typing import Any, Sequence

from orbit.memory.config import MemoryConfig
from orbit.memory.errors import MemoryModelError, MemoryValidationError


def _exception_detail(exc: Exception) -> str:
    message = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {message}" if message else name


def _configure_transformers_loading() -> None:
    """Use deterministic, memory-bounded model loading by default.

    Transformers v5 materializes checkpoint tensors asynchronously unless
    HF_DEACTIVATE_ASYNC_LOAD is enabled. Orbit's memory models are large enough
    that the async path can create a high peak-memory/native-loading failure on
    Windows, so prefer sequential materialization. Respect an explicit user
    override if they set the variable themselves.
    """
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")


class MemoryModels:
    """Lazy local-only embedding and reranking models."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._embedder: Any | None = None
        self._reranker: Any | None = None
        self._embedder_lock = Lock()
        self._reranker_lock = Lock()

    def _get_embedder(self) -> Any:
        if self._embedder is not None:
            return self._embedder

        with self._embedder_lock:
            if self._embedder is not None:
                return self._embedder
            try:
                _configure_transformers_loading()
                from sentence_transformers import SentenceTransformer

                kwargs: dict[str, Any] = {"local_files_only": True}
                if self.config.device is not None:
                    kwargs["device"] = self.config.device
                model = SentenceTransformer(self.config.embedding_model, **kwargs)
            except Exception as exc:
                raise MemoryModelError(
                    "Could not load local embedding model "
                    f"'{self.config.embedding_model}'. Ensure it exists in the local "
                    "Hugging Face cache. "
                    f"Underlying error: {_exception_detail(exc)}"
                ) from exc

            dimension = model.get_embedding_dimension()
            if dimension != self.config.vector_size:
                raise MemoryModelError(
                    f"Embedding model '{self.config.embedding_model}' has dimension "
                    f"{dimension}; expected {self.config.vector_size}."
                )

            self._embedder = model
            return model

    def _get_reranker(self) -> Any:
        if self._reranker is not None:
            return self._reranker

        with self._reranker_lock:
            if self._reranker is not None:
                return self._reranker
            try:
                _configure_transformers_loading()
                from sentence_transformers import CrossEncoder

                kwargs: dict[str, Any] = {"local_files_only": True}
                if self.config.device is not None:
                    kwargs["device"] = self.config.device
                model = CrossEncoder(self.config.reranker_model, **kwargs)
            except Exception as exc:
                raise MemoryModelError(
                    "Could not load local reranker model "
                    f"'{self.config.reranker_model}'. Ensure it exists in the local "
                    "Hugging Face cache. "
                    f"Underlying error: {_exception_detail(exc)}"
                ) from exc

            self._reranker = model
            return model

    @staticmethod
    def _token_count(model: Any, text: str) -> int:
        tokenizer = model.tokenizer
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        except Exception as exc:
            raise MemoryModelError(
                f"Could not tokenize memory input: {_exception_detail(exc)}"
            ) from exc

        if not isinstance(encoded, Mapping):
            raise MemoryModelError(
                "Embedding tokenizer returned an unsupported encoding object."
            )

        input_ids = encoded.get("input_ids")
        if input_ids is None:
            raise MemoryModelError("Embedding tokenizer did not return input_ids.")

        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            input_ids = input_ids[0]

        try:
            return len(input_ids)
        except TypeError as exc:
            raise MemoryModelError(
                "Embedding tokenizer returned invalid input_ids."
            ) from exc

    def _validate_length(self, text: str, *, max_tokens: int, kind: str) -> None:
        model = self._get_embedder()
        count = self._token_count(model, text)
        if count > max_tokens:
            raise MemoryValidationError(
                f"{kind} is too long ({count} tokens; maximum {max_tokens}). "
                "Use a shorter, self-contained value."
            )

    def validate_memory(self, text: str) -> None:
        self._validate_length(
            f"passage: {text}",
            max_tokens=self.config.max_memory_tokens,
            kind="Memory",
        )

    def embed_memory(self, text: str) -> list[float]:
        model = self._get_embedder()
        try:
            vector = model.encode(
                f"passage: {text}",
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vector.tolist()
        except Exception as exc:
            raise MemoryModelError(
                f"Could not embed memory text: {_exception_detail(exc)}"
            ) from exc

    def embed_query(self, query: str) -> list[float]:
        model = self._get_embedder()
        self._validate_length(
            f"query: {query}",
            max_tokens=self.config.max_query_tokens,
            kind="Memory query",
        )
        try:
            vector = model.encode(
                f"query: {query}",
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vector.tolist()
        except Exception as exc:
            raise MemoryModelError(
                f"Could not embed memory query: {_exception_detail(exc)}"
            ) from exc

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []

        model = self._get_reranker()
        pairs = [(query, passage) for passage in passages]
        try:
            scores = model.predict(
                pairs,
                batch_size=self.config.rerank_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise MemoryModelError(
                f"Could not rerank memory candidates: {_exception_detail(exc)}"
            ) from exc

        if getattr(scores, "ndim", 1) > 1:
            scores = scores.reshape(-1)
        return [float(score) for score in scores]

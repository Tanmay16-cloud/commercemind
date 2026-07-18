from __future__ import annotations

import math
from collections.abc import Sequence
from hashlib import blake2b
from typing import Any, Protocol, cast

import numpy as np

from commercemind.retrieval.baseline import tokenize


class TextEmbedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Convert a batch of text strings into numeric vectors."""


class HashingTextEmbedder:
    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._dimensions), dtype=np.float32)

        for row_index, text in enumerate(texts):
            for token in tokenize(text):
                column_index = _stable_hash(token) % self._dimensions
                vectors[row_index, column_index] += 1.0

        return l2_normalize(vectors)

    @property
    def backend(self) -> str:
        return "hashing"

    @property
    def model_name(self) -> str:
        return f"blake2b-token-hash-{self._dimensions}d"


class SentenceTransformerTextEmbedder:
    """Sentence Transformer adapter with lazy model loading."""

    def __init__(self, model_name: str) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be blank")
        self._model_name = model_name
        self._model: Any | None = None

    @property
    def backend(self) -> str:
        return "sentence_transformer"

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load_model()
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model


def create_text_embedder(
    backend: str,
    *,
    model_name: str,
    hashing_dimensions: int = 384,
) -> TextEmbedder:
    normalized_backend = backend.strip().lower().replace("-", "_")
    if normalized_backend == "hashing":
        return HashingTextEmbedder(dimensions=hashing_dimensions)
    if normalized_backend in {"sentence_transformer", "sentence_transformers"}:
        return SentenceTransformerTextEmbedder(model_name=model_name)
    raise ValueError(
        f"unknown embedding backend '{backend}'. Expected: hashing or sentence_transformer"
    )


def embedder_identity(embedder: TextEmbedder) -> tuple[str, str | None]:
    backend = cast(str, getattr(embedder, "backend", "custom"))
    model_name = cast(str | None, getattr(embedder, "model_name", None))
    return backend, model_name


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    return vectors / safe_norms


def cosine_similarity(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    if query_vector.ndim != 1:
        raise ValueError("query_vector must be one-dimensional")
    if document_vectors.ndim != 2:
        raise ValueError("document_vectors must be two-dimensional")
    if document_vectors.shape[1] != query_vector.shape[0]:
        raise ValueError("query_vector and document_vectors must have the same width")

    normalized_query = l2_normalize(query_vector.reshape(1, -1))[0]
    normalized_documents = l2_normalize(document_vectors)
    return cast(np.ndarray, normalized_documents @ normalized_query)


def weighted_text(fields: Sequence[tuple[str | None, float]]) -> str:
    parts: list[str] = []

    for value, weight in fields:
        if not value:
            continue
        repetitions = max(1, math.ceil(weight))
        parts.extend([value] * repetitions)

    return " ".join(parts)


def _stable_hash(value: str) -> int:
    digest = blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)

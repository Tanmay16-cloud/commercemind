from __future__ import annotations

import math
from hashlib import blake2b
from typing import Protocol, Sequence

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
    return normalized_documents @ normalized_query


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

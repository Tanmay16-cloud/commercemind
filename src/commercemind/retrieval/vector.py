from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from commercemind.retrieval.baseline import RetrievalCandidate, tokenize
from commercemind.retrieval.embeddings import (
    HashingTextEmbedder,
    TextEmbedder,
    cosine_similarity,
    weighted_text,
)


@dataclass(frozen=True)
class VectorDocument:
    item_id: str
    title: str
    text: str


class VectorRetriever:
    def __init__(self, products: pl.DataFrame, embedder: TextEmbedder | None = None) -> None:
        self._embedder = embedder or HashingTextEmbedder()
        self._documents = [_product_document(row) for row in products.iter_rows(named=True)]
        self._document_vectors = self._embedder.encode(
            [document.text for document in self._documents]
        )

    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        if top_k <= 0 or not tokenize(query):
            return []

        query_vector = self._embedder.encode([query])[0]
        scores = cosine_similarity(query_vector, self._document_vectors)
        ranked_indexes = np.argsort(-scores, kind="mergesort")

        candidates: list[RetrievalCandidate] = []

        for index in ranked_indexes:
            score = float(scores[index])
            if score <= 0:
                continue

            document = self._documents[int(index)]
            candidates.append(
                RetrievalCandidate(
                    item_id=document.item_id,
                    title=document.title,
                    score=score,
                )
            )

            if len(candidates) == top_k:
                break

        return candidates


def _product_document(row: dict[str, object]) -> VectorDocument:
    title = str(row["title"])
    text = weighted_text(
        [
            (_optional_text(row, "title"), 3.0),
            (_optional_text(row, "category"), 2.0),
            (_optional_text(row, "brand"), 1.0),
            (_optional_text(row, "description"), 1.0),
        ]
    )

    return VectorDocument(
        item_id=str(row["item_id"]),
        title=title,
        text=text,
    )


def _optional_text(row: dict[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    return str(value)

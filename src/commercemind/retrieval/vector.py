from __future__ import annotations

import polars as pl

from commercemind.retrieval.baseline import RetrievalCandidate, tokenize
from commercemind.retrieval.embeddings import (
    HashingTextEmbedder,
    TextEmbedder,
    create_text_embedder,
)
from commercemind.retrieval.vector_index import (
    FaissVectorIndex,
    VectorIndexError,
    build_faiss_vector_index,
)


class VectorRetriever:
    def __init__(
        self,
        products: pl.DataFrame,
        embedder: TextEmbedder | None = None,
        vector_index: FaissVectorIndex | None = None,
        embedding_backend: str | None = None,
        embedding_model_name: str | None = None,
    ) -> None:
        if vector_index is not None:
            vector_index.validate_catalog(products)
            self._embedder = embedder or _embedder_for_index(
                vector_index,
                configured_backend=embedding_backend,
                configured_model=embedding_model_name,
            )
            self._index = vector_index
            return

        self._embedder = embedder or (
            create_text_embedder(
                embedding_backend,
                model_name=embedding_model_name or "sentence-transformers/all-MiniLM-L6-v2",
            )
            if embedding_backend is not None
            else HashingTextEmbedder()
        )
        self._index = build_faiss_vector_index(products, self._embedder)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        if top_k <= 0 or not tokenize(query):
            return []

        query_vector = self._embedder.encode([query])[0]
        return [
            RetrievalCandidate(
                item_id=result.document.item_id,
                title=result.document.title,
                score=result.score,
            )
            for result in self._index.search(query_vector, top_k)
        ]


def _embedder_for_index(
    index: FaissVectorIndex,
    *,
    configured_backend: str | None,
    configured_model: str | None,
) -> TextEmbedder:
    backend = index.embedding_backend
    if backend == "custom":
        raise VectorIndexError("a custom vector index requires an explicit query embedder")

    if configured_backend is not None:
        normalized_backend = configured_backend.strip().lower().replace("-", "_")
        if normalized_backend != backend:
            raise VectorIndexError(
                "configured embedding backend does not match the vector index artifact"
            )

    if (
        backend == "sentence_transformer"
        and configured_model is not None
        and index.embedding_model != configured_model
    ):
        raise VectorIndexError(
            "configured embedding model does not match the vector index artifact"
        )

    return create_text_embedder(
        backend,
        model_name=index.embedding_model or configured_model or "",
        hashing_dimensions=index.dimensions,
    )

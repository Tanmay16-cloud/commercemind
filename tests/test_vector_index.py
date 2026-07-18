from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest

from commercemind.retrieval.baseline import tokenize
from commercemind.retrieval.embeddings import l2_normalize
from commercemind.retrieval.vector import VectorRetriever
from commercemind.retrieval.vector_index import (
    INDEX_FILENAME,
    METADATA_FILENAME,
    FaissVectorIndex,
    VectorIndexError,
    build_faiss_vector_index,
)


class ToySemanticEmbedder:
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 2), dtype=np.float32)

        for row_index, text in enumerate(texts):
            tokens = set(tokenize(text))
            if tokens & {"running", "shoes", "sneakers", "footwear"}:
                vectors[row_index, 0] = 1.0
            if tokens & {"office", "formal", "shirt", "work"}:
                vectors[row_index, 1] = 1.0

        return l2_normalize(vectors)


def test_faiss_vector_index_returns_semantic_matches() -> None:
    products = _products()
    embedder = ToySemanticEmbedder()
    index = build_faiss_vector_index(products, embedder)

    query_vector = embedder.encode(["comfortable sneakers"])[0]
    results = index.search(query_vector, top_k=2)

    assert [result.document.item_id for result in results] == ["A1"]
    assert results[0].score > 0


def test_faiss_vector_index_round_trips_to_disk(tmp_path) -> None:
    products = _products()
    embedder = ToySemanticEmbedder()
    index = build_faiss_vector_index(products, embedder)

    index.save(tmp_path)
    loaded = FaissVectorIndex.load(tmp_path)

    query_vector = embedder.encode(["formal work shirt"])[0]
    results = loaded.search(query_vector, top_k=2)

    assert (tmp_path / INDEX_FILENAME).exists()
    assert (tmp_path / METADATA_FILENAME).exists()
    assert loaded.size == 2
    assert loaded.dimensions == 2
    assert [result.document.item_id for result in results] == ["A2"]


def test_vector_retriever_can_use_prebuilt_faiss_index() -> None:
    products = _products()
    embedder = ToySemanticEmbedder()
    index = build_faiss_vector_index(products, embedder)
    retriever = VectorRetriever(products, embedder=embedder, vector_index=index)

    candidates = retriever.retrieve("comfortable sneakers", top_k=2)

    assert [candidate.item_id for candidate in candidates] == ["A1"]


def test_vector_retriever_rejects_index_from_different_catalog() -> None:
    products = _products()
    index = build_faiss_vector_index(products, ToySemanticEmbedder())
    changed_products = products.with_columns(pl.lit("Changed title").alias("title"))

    with pytest.raises(VectorIndexError, match="catalog fingerprint"):
        VectorRetriever(
            changed_products,
            embedder=ToySemanticEmbedder(),
            vector_index=index,
        )


def test_loading_missing_vector_index_raises_clear_error(tmp_path) -> None:
    with pytest.raises(VectorIndexError, match="missing vector index artifact"):
        FaissVectorIndex.load(tmp_path)


def _products() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "item_id": ["A1", "A2"],
            "title": ["Running Shoes", "Office Shirt"],
            "category": ["Footwear", "Apparel"],
            "brand": ["Nike", "Acme"],
            "description": ["Daily training shoes", "Formal shirt for work"],
            "price": [2999.0, 1499.0],
        }
    )

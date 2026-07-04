from typing import Sequence

import numpy as np
import polars as pl

from commercemind.retrieval.baseline import tokenize
from commercemind.retrieval.embeddings import l2_normalize
from commercemind.retrieval.vector import VectorRetriever


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


def test_vector_retriever_ranks_semantically_similar_products() -> None:
    products = pl.DataFrame(
        {
            "item_id": ["A1", "A2"],
            "title": ["Running Shoes", "Office Shirt"],
            "category": ["Footwear", "Apparel"],
            "brand": ["Nike", "Acme"],
            "description": ["Daily training shoes", "Formal shirt for work"],
            "price": [2999.0, 1499.0],
        }
    )

    retriever = VectorRetriever(products, embedder=ToySemanticEmbedder())
    candidates = retriever.retrieve("comfortable sneakers", top_k=2)

    assert [candidate.item_id for candidate in candidates] == ["A1"]
    assert candidates[0].score > 0


def test_vector_retriever_returns_empty_list_for_empty_query() -> None:
    products = pl.DataFrame(
        {
            "item_id": ["A1"],
            "title": ["Running Shoes"],
            "category": ["Footwear"],
            "brand": ["Nike"],
            "description": ["Daily training shoes"],
        }
    )

    retriever = VectorRetriever(products, embedder=ToySemanticEmbedder())

    assert retriever.retrieve("", top_k=5) == []

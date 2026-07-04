import polars as pl

from commercemind.retrieval.baseline import LexicalRetriever
from commercemind.retrieval.hybrid import HybridRetriever, SearchRetriever
from commercemind.retrieval.vector import VectorRetriever
from commercemind.schemas import ItemResult, SearchRequest, SearchResponse


class SearchService:
    def __init__(
        self,
        products: pl.DataFrame | None = None,
        retriever: SearchRetriever | None = None,
    ) -> None:
        self._products = products if products is not None else _default_products()
        self._retriever = retriever or HybridRetriever(
            [
                LexicalRetriever(self._products),
                VectorRetriever(self._products),
            ]
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        normalized_query = request.query.strip()
        candidates = self._retriever.retrieve(normalized_query, request.top_k)
        results = [
            ItemResult(
                item_id=candidate.item_id,
                title=candidate.title,
                score=candidate.score,
            )
            for candidate in candidates
        ]

        return SearchResponse(query=normalized_query, results=results)


def _default_products() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "item_id": ["sku-running-shoes-001", "sku-gym-shirt-001", "sku-headphones-001"],
            "title": ["Running Shoes", "Gym T-Shirt", "Wireless Headphones"],
            "category": ["Footwear", "Apparel", "Electronics"],
            "brand": ["StrideLab", "StrideLab", "NorthAudio"],
            "description": [
                "Lightweight shoes for road running and daily training.",
                "Breathable shirt for workouts and gym sessions.",
                "Bluetooth headphones for music and calls.",
            ],
            "price": [2999.0, 999.0, 2499.0],
        }
    )

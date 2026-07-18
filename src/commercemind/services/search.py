import polars as pl

from commercemind.ranking.rankers import LinearProductRanker, ProductRanker
from commercemind.retrieval.baseline import LexicalRetriever
from commercemind.retrieval.hybrid import HybridRetriever, SearchRetriever
from commercemind.retrieval.vector import VectorRetriever
from commercemind.retrieval.vector_index import FaissVectorIndex
from commercemind.schemas import ItemResult, SearchRequest, SearchResponse
from commercemind.services.sample_data import default_products


class SearchService:
    def __init__(
        self,
        products: pl.DataFrame | None = None,
        retriever: SearchRetriever | None = None,
        ranker: ProductRanker | None = None,
        vector_index: FaissVectorIndex | None = None,
        embedding_backend: str | None = None,
        embedding_model_name: str | None = None,
        candidate_pool_multiplier: int = 5,
    ) -> None:
        if candidate_pool_multiplier <= 0:
            raise ValueError("candidate_pool_multiplier must be positive")

        self._products = products if products is not None else default_products()
        self._retriever = retriever or HybridRetriever(
            [
                LexicalRetriever(self._products),
                VectorRetriever(
                    self._products,
                    vector_index=vector_index,
                    embedding_backend=embedding_backend,
                    embedding_model_name=embedding_model_name,
                ),
            ]
        )
        self._ranker = ranker or LinearProductRanker(self._products)
        self._candidate_pool_multiplier = candidate_pool_multiplier

    def search(self, request: SearchRequest) -> SearchResponse:
        normalized_query = request.query.strip()
        candidate_pool_size = request.top_k * self._candidate_pool_multiplier
        candidates = self._retriever.retrieve(normalized_query, candidate_pool_size)
        ranked_candidates = self._ranker.rank(normalized_query, candidates, request.top_k)
        results = [
            ItemResult(
                item_id=candidate.item_id,
                title=candidate.title,
                score=candidate.score,
            )
            for candidate in ranked_candidates
        ]

        return SearchResponse(query=normalized_query, results=results)

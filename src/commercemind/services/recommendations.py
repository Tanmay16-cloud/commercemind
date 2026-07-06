import polars as pl

from commercemind.ranking.rankers import LinearProductRanker, ProductRanker
from commercemind.recommendation.candidates import PersonalizedCandidateGenerator
from commercemind.schemas import ItemResult, RecommendationRequest, RecommendationResponse
from commercemind.services.sample_data import default_interactions, default_products


class RecommendationService:
    def __init__(
        self,
        products: pl.DataFrame | None = None,
        interactions: pl.DataFrame | None = None,
        ranker: ProductRanker | None = None,
        candidate_pool_multiplier: int = 5,
    ) -> None:
        if candidate_pool_multiplier <= 0:
            raise ValueError("candidate_pool_multiplier must be positive")

        self._products = products if products is not None else default_products()
        self._interactions = interactions if interactions is not None else default_interactions()
        self._candidate_generator = PersonalizedCandidateGenerator(
            self._products,
            self._interactions,
        )
        self._ranker = ranker or LinearProductRanker(self._products)
        self._candidate_pool_multiplier = candidate_pool_multiplier

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        candidate_pool_size = request.top_k * self._candidate_pool_multiplier
        profile = self._candidate_generator.build_user_profile(request.user_id)
        candidates = self._candidate_generator.generate(request.user_id, candidate_pool_size)
        ranked_candidates = self._ranker.rank(profile.as_query(), candidates, request.top_k)

        results = [
            ItemResult(
                item_id=candidate.item_id,
                title=candidate.title,
                score=candidate.score,
            )
            for candidate in ranked_candidates
        ]

        return RecommendationResponse(user_id=request.user_id, results=results)

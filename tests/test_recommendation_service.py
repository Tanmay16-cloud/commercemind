import polars as pl

from commercemind.retrieval.baseline import RetrievalCandidate
from commercemind.schemas import RecommendationRequest
from commercemind.services.recommendations import RecommendationService


class SpyRanker:
    def __init__(self) -> None:
        self.query = ""
        self.top_k = 0

    def rank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        self.query = query
        self.top_k = top_k
        return candidates[:top_k]


def test_recommendation_service_returns_personalized_results() -> None:
    products = pl.DataFrame(
        {
            "item_id": ["A1", "A2", "A3"],
            "title": ["Road Running Shoes", "Trail Running Shoes", "Wireless Headphones"],
            "category": ["Footwear", "Footwear", "Electronics"],
            "brand": ["StrideLab", "StrideLab", "NorthAudio"],
            "description": ["Road shoe", "Trail shoe", "Audio gear"],
            "price": [2999.0, 3499.0, 2499.0],
        }
    )
    interactions = pl.DataFrame(
        {
            "user_id": ["U1"],
            "item_id": ["A1"],
            "event_type": ["purchase"],
            "timestamp_ms": [1],
        }
    )
    ranker = SpyRanker()
    service = RecommendationService(
        products=products,
        interactions=interactions,
        ranker=ranker,
    )

    response = service.recommend(RecommendationRequest(user_id="U1", top_k=1))

    assert response.user_id == "U1"
    assert response.results[0].item_id == "A2"
    assert "Footwear" in ranker.query
    assert ranker.top_k == 1

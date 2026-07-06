import polars as pl

from commercemind.ranking.rankers import LinearProductRanker, RankingWeights
from commercemind.retrieval.baseline import RetrievalCandidate


def test_linear_product_ranker_boosts_exact_title_match() -> None:
    products = pl.DataFrame(
        {
            "item_id": ["A1", "A2"],
            "title": ["Running Shoes", "Gym Shirt"],
            "category": ["Footwear", "Apparel"],
            "brand": ["StrideLab", "StrideLab"],
            "description": ["Lightweight road running shoes", "Breathable workout shirt"],
            "price": [2999.0, 999.0],
        }
    )
    candidates = [
        RetrievalCandidate(item_id="A2", title="Gym Shirt", score=0.9),
        RetrievalCandidate(item_id="A1", title="Running Shoes", score=0.8),
    ]
    ranker = LinearProductRanker(
        products,
        weights=RankingWeights(
            retrieval_score=0.1,
            title_match=2.0,
            category_match=0.0,
            brand_match=0.0,
            description_match=0.0,
            price_intent_match=0.0,
        ),
    )

    ranked = ranker.rank("running shoes", candidates, top_k=2)

    assert [candidate.item_id for candidate in ranked] == ["A1", "A2"]
    assert ranked[0].score > ranked[1].score


def test_linear_product_ranker_respects_top_k() -> None:
    products = pl.DataFrame(
        {
            "item_id": ["A1", "A2"],
            "title": ["Running Shoes", "Gym Shirt"],
            "category": ["Footwear", "Apparel"],
            "brand": ["StrideLab", "StrideLab"],
            "description": ["Lightweight road running shoes", "Breathable workout shirt"],
            "price": [2999.0, 999.0],
        }
    )
    candidates = [
        RetrievalCandidate(item_id="A1", title="Running Shoes", score=0.8),
        RetrievalCandidate(item_id="A2", title="Gym Shirt", score=0.7),
    ]

    ranker = LinearProductRanker(products)
    ranked = ranker.rank("running shoes", candidates, top_k=1)

    assert len(ranked) == 1

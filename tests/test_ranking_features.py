import polars as pl

from commercemind.ranking.features import ProductFeatureStore
from commercemind.retrieval.baseline import RetrievalCandidate


def test_product_feature_store_builds_text_match_features() -> None:
    products = pl.DataFrame(
        {
            "item_id": ["A1"],
            "title": ["Running Shoes"],
            "category": ["Footwear"],
            "brand": ["StrideLab"],
            "description": ["Lightweight shoes for road running"],
            "price": [2999.0],
        }
    )
    feature_store = ProductFeatureStore(products)

    features = feature_store.build_features(
        "running shoes",
        RetrievalCandidate(item_id="A1", title="Running Shoes", score=0.5),
    )

    assert features.retrieval_score == 0.5
    assert features.title_match == 1.0
    assert features.description_match == 1.0
    assert features.category_match == 0.0


def test_product_feature_store_detects_budget_price_intent() -> None:
    products = pl.DataFrame(
        {
            "item_id": ["A1", "A2"],
            "title": ["Budget Shirt", "Premium Jacket"],
            "category": ["Apparel", "Apparel"],
            "brand": ["Acme", "Acme"],
            "description": ["Basic shirt", "Leather jacket"],
            "price": [500.0, 5000.0],
        }
    )
    feature_store = ProductFeatureStore(products)

    budget_features = feature_store.build_features(
        "cheap apparel",
        RetrievalCandidate(item_id="A1", title="Budget Shirt", score=0.5),
    )
    premium_features = feature_store.build_features(
        "cheap apparel",
        RetrievalCandidate(item_id="A2", title="Premium Jacket", score=0.5),
    )

    assert budget_features.price_intent_match > premium_features.price_intent_match

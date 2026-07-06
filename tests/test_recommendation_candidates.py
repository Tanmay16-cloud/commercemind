import polars as pl

from commercemind.recommendation.candidates import PersonalizedCandidateGenerator


def test_personalized_candidate_generator_scores_unseen_matching_products() -> None:
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

    generator = PersonalizedCandidateGenerator(products, interactions)
    candidates = generator.generate("U1", top_k=2)

    assert candidates[0].item_id == "A2"
    assert "A1" not in [candidate.item_id for candidate in candidates]


def test_personalized_candidate_generator_uses_popularity_for_new_users() -> None:
    products = pl.DataFrame(
        {
            "item_id": ["A1", "A2"],
            "title": ["Running Shoes", "Wireless Headphones"],
            "category": ["Footwear", "Electronics"],
            "brand": ["StrideLab", "NorthAudio"],
            "description": ["Road shoe", "Audio gear"],
            "price": [2999.0, 2499.0],
        }
    )
    interactions = pl.DataFrame(
        {
            "user_id": ["U1", "U2", "U2"],
            "item_id": ["A1", "A2", "A2"],
            "event_type": ["click", "click", "purchase"],
            "timestamp_ms": [1, 2, 3],
        }
    )

    generator = PersonalizedCandidateGenerator(products, interactions)
    candidates = generator.generate("new-user", top_k=1)

    assert candidates[0].item_id == "A2"

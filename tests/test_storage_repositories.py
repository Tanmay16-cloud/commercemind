import polars as pl

from commercemind.schemas import RecommendationRequest
from commercemind.services.recommendations import RecommendationService
from commercemind.storage.database import build_engine, create_schema, create_session_factory
from commercemind.storage.repositories import InteractionRepository, ProductRepository
from commercemind.storage.seed import seed_catalog


def test_product_and_interaction_repositories_round_trip_dataframes() -> None:
    engine = build_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)

    products = pl.DataFrame(
        {
            "item_id": ["A1", "A2"],
            "title": ["Running Shoes", "Wireless Headphones"],
            "category": ["Footwear", "Electronics"],
            "brand": ["StrideLab", "NorthAudio"],
            "description": ["Road running shoe", "Bluetooth headphones"],
            "price": [2999.0, 2499.0],
        }
    )
    interactions = pl.DataFrame(
        {
            "user_id": ["U1", "U2"],
            "item_id": ["A1", "A2"],
            "event_type": ["purchase", "click"],
            "timestamp_ms": [1, 2],
        }
    )

    with session_factory() as session:
        ProductRepository(session).replace_all(products)
        InteractionRepository(session).replace_all(interactions)
        session.commit()

        stored_products = ProductRepository(session).list_all()
        stored_interactions = InteractionRepository(session).list_all()

    assert stored_products["item_id"].to_list() == ["A1", "A2"]
    assert stored_products["price"].to_list() == [2999.0, 2499.0]
    assert stored_interactions["event_type"].to_list() == ["purchase", "click"]


def test_seeded_database_can_feed_recommendation_service() -> None:
    engine = build_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        seed_catalog(session)
        products = ProductRepository(session).list_all()
        interactions = InteractionRepository(session).list_all()

    service = RecommendationService(products=products, interactions=interactions)
    response = service.recommend(RecommendationRequest(user_id="user-runner", top_k=1))

    assert response.results[0].item_id == "sku-trail-shoes-001"

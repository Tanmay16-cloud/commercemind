import polars as pl

from commercemind.data.transform import normalize_interactions, normalize_products


def test_normalize_products_maps_supported_columns() -> None:
    products = pl.DataFrame(
        {
            "parent_asin": ["A1"],
            "title": ["  Running Shoes  "],
            "main_category": ["Footwear"],
            "store": ["Nike"],
            "price": [2999.0],
        }
    )

    normalized = normalize_products(products)

    assert normalized.columns == ["item_id", "title", "category", "brand", "description", "price"]
    assert normalized.row(0) == ("A1", "Running Shoes", "Footwear", "Nike", None, 2999.0)


def test_normalize_interactions_maps_supported_columns() -> None:
    interactions = pl.DataFrame(
        {
            "user_id": ["U1"],
            "asin": ["A1"],
            "timestamp": [1720000000000],
            "rating": [4.0],
        }
    )

    normalized = normalize_interactions(interactions)

    assert normalized.columns == ["user_id", "item_id", "event_type", "timestamp_ms", "rating"]
    assert normalized.row(0) == ("U1", "A1", "rating", 1720000000000, 4.0)

import polars as pl


def default_products() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "item_id": [
                "sku-running-shoes-001",
                "sku-trail-shoes-001",
                "sku-gym-shirt-001",
                "sku-headphones-001",
            ],
            "title": [
                "Running Shoes",
                "Trail Running Shoes",
                "Gym T-Shirt",
                "Wireless Headphones",
            ],
            "category": ["Footwear", "Footwear", "Apparel", "Electronics"],
            "brand": ["StrideLab", "StrideLab", "StrideLab", "NorthAudio"],
            "description": [
                "Lightweight shoes for road running and daily training.",
                "Durable shoes for trails, grip, and outdoor running.",
                "Breathable shirt for workouts and gym sessions.",
                "Bluetooth headphones for music and calls.",
            ],
            "price": [2999.0, 3499.0, 999.0, 2499.0],
        }
    )


def default_interactions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": ["user-runner", "user-runner", "user-audio", "user-audio"],
            "item_id": [
                "sku-running-shoes-001",
                "sku-gym-shirt-001",
                "sku-headphones-001",
                "sku-headphones-001",
            ],
            "event_type": ["purchase", "click", "click", "purchase"],
            "timestamp_ms": [1, 2, 3, 4],
        }
    )

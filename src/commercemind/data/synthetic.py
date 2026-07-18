from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import polars as pl

from commercemind.data.datasets import DatasetPaths, get_dataset_paths

DEFAULT_SYNTHETIC_DATASET_NAME = "synthetic_large"

class CategoryTemplate(TypedDict):
    types: list[str]
    brands: list[str]
    base_price: float


CATEGORY_TEMPLATES: dict[str, CategoryTemplate] = {
    "Footwear": {
        "types": ["Running Shoes", "Trail Shoes", "Walking Sneakers", "Training Shoes"],
        "brands": ["StrideLab", "TrailForge", "WalkWell", "CoreStep"],
        "base_price": 2499.0,
    },
    "Apparel": {
        "types": ["Training T-Shirt", "Running Shorts", "Fleece Hoodie", "Yoga Jacket"],
        "brands": ["CoreWear", "FlexFit", "MotionCraft", "UrbanLayer"],
        "base_price": 899.0,
    },
    "Electronics": {
        "types": ["Wireless Headphones", "Sport Earbuds", "Bluetooth Speaker", "Fitness Watch"],
        "brands": ["NorthAudio", "PulseTrack", "SoundPeak", "WaveNest"],
        "base_price": 2499.0,
    },
    "Fitness": {
        "types": ["Yoga Mat", "Water Bottle", "Resistance Bands", "Foam Roller"],
        "brands": ["ZenFlex", "HydraGo", "LiftLoop", "RecoverPro"],
        "base_price": 699.0,
    },
    "Home": {
        "types": ["Desk Lamp", "Storage Basket", "Cotton Bedsheet", "Travel Mug"],
        "brands": ["HomeNest", "BrightRoom", "CozyCraft", "DailyEase"],
        "base_price": 599.0,
    },
}

DESCRIPTORS = [
    "Lightweight",
    "Premium",
    "Budget",
    "Durable",
    "Compact",
    "Breathable",
    "Waterproof",
    "Pro",
]

EVENT_TYPES = ["view", "click", "add_to_cart", "purchase"]
EVENT_WEIGHTS = [0.45, 0.30, 0.15, 0.10]


@dataclass(frozen=True)
class SyntheticDatasetSpec:
    dataset_name: str = DEFAULT_SYNTHETIC_DATASET_NAME
    product_count: int = 1000
    user_count: int = 100
    interaction_count: int = 5000
    seed: int = 42


@dataclass(frozen=True)
class SyntheticDatasetResult:
    dataset_name: str
    products_path: Path
    interactions_path: Path
    product_count: int
    user_count: int
    interaction_count: int


def generate_synthetic_products(product_count: int, *, seed: int = 42) -> pl.DataFrame:
    if product_count <= 0:
        raise ValueError("product_count must be positive")

    rng = random.Random(seed)
    categories = list(CATEGORY_TEMPLATES)
    rows: list[dict[str, object]] = []

    for index in range(product_count):
        category = categories[index % len(categories)]
        template = CATEGORY_TEMPLATES[category]
        product_type = rng.choice(template["types"])
        brand = rng.choice(template["brands"])
        descriptor = rng.choice(DESCRIPTORS)
        price = _synthetic_price(float(template["base_price"]), rng)

        rows.append(
            {
                "item_id": f"synthetic-item-{index:06d}",
                "title": f"{descriptor} {brand} {product_type}",
                "category": category,
                "brand": brand,
                "description": (
                    f"{descriptor.lower()} {product_type.lower()} for "
                    f"{category.lower()} shoppers who prefer {brand} products."
                ),
                "price": price,
            }
        )

    return pl.DataFrame(rows)


def generate_synthetic_interactions(
    products: pl.DataFrame,
    *,
    user_count: int = 100,
    interaction_count: int = 5000,
    seed: int = 42,
) -> pl.DataFrame:
    if products.is_empty():
        raise ValueError("products must not be empty")
    if user_count <= 0:
        raise ValueError("user_count must be positive")
    if interaction_count <= 0:
        raise ValueError("interaction_count must be positive")

    rng = random.Random(seed + 1)
    product_rows = [row for row in products.iter_rows(named=True)]
    products_by_category = _products_by_category(product_rows)
    categories = sorted(products_by_category)
    rows: list[dict[str, object]] = []

    for index in range(interaction_count):
        user_index = rng.randrange(user_count)
        preferred_category = categories[user_index % len(categories)]
        product_pool = (
            products_by_category[preferred_category]
            if rng.random() < 0.75
            else product_rows
        )
        product = rng.choice(product_pool)

        rows.append(
            {
                "user_id": f"synthetic-user-{user_index:05d}",
                "item_id": str(product["item_id"]),
                "event_type": rng.choices(EVENT_TYPES, weights=EVENT_WEIGHTS, k=1)[0],
                "timestamp_ms": 1_700_000_000_000 + index * 1000,
            }
        )

    return pl.DataFrame(rows)


def build_synthetic_dataset(
    spec: SyntheticDatasetSpec,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    products = generate_synthetic_products(spec.product_count, seed=spec.seed)
    interactions = generate_synthetic_interactions(
        products,
        user_count=spec.user_count,
        interaction_count=spec.interaction_count,
        seed=spec.seed,
    )
    return products, interactions


def materialize_synthetic_dataset(
    spec: SyntheticDatasetSpec,
    *,
    processed_dir: Path | None = None,
) -> SyntheticDatasetResult:
    paths = _dataset_paths(spec.dataset_name, processed_dir)
    products, interactions = build_synthetic_dataset(spec)

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    products.write_parquet(paths.products_path)
    interactions.write_parquet(paths.interactions_path)

    return SyntheticDatasetResult(
        dataset_name=spec.dataset_name,
        products_path=paths.products_path,
        interactions_path=paths.interactions_path,
        product_count=products.height,
        user_count=spec.user_count,
        interaction_count=interactions.height,
    )


def _dataset_paths(dataset_name: str, processed_dir: Path | None) -> DatasetPaths:
    if processed_dir is None:
        return get_dataset_paths(dataset_name)

    return DatasetPaths(
        dataset_name=dataset_name,
        raw_dir=processed_dir.parent / "raw" / dataset_name,
        processed_dir=processed_dir,
    )


def _products_by_category(
    product_rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}

    for row in product_rows:
        grouped.setdefault(str(row["category"]), []).append(row)

    return grouped


def _synthetic_price(base_price: float, rng: random.Random) -> float:
    multiplier = rng.uniform(0.65, 1.85)
    return round(base_price * multiplier, 2)

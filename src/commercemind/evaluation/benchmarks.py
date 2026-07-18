from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from commercemind.evaluation.recommendation import RecommendationEvaluationCase
from commercemind.evaluation.retrieval import RetrievalEvaluationCase
from commercemind.services.sample_data import default_interactions, default_products


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    products: pl.DataFrame
    interactions: pl.DataFrame
    ranking_training_cases: list[RetrievalEvaluationCase]
    search_cases: list[RetrievalEvaluationCase]
    recommendation_cases: list[RecommendationEvaluationCase]


def build_benchmark_dataset(name: str) -> BenchmarkDataset:
    if name == "demo":
        return build_demo_benchmark_dataset()
    if name == "sample":
        return build_sample_benchmark_dataset()
    raise ValueError(f"unknown benchmark dataset: {name}")


def build_demo_benchmark_dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        name="demo",
        products=default_products(),
        interactions=default_interactions(),
        ranking_training_cases=[
            RetrievalEvaluationCase(
                query="road running footwear",
                relevant_item_ids={"sku-running-shoes-001", "sku-trail-shoes-001"},
            ),
            RetrievalEvaluationCase(
                query="workout top",
                relevant_item_ids={"sku-gym-shirt-001"},
            ),
            RetrievalEvaluationCase(
                query="bluetooth audio",
                relevant_item_ids={"sku-headphones-001"},
            ),
        ],
        search_cases=[
            RetrievalEvaluationCase(
                query="running shoes",
                relevant_item_ids={"sku-running-shoes-001", "sku-trail-shoes-001"},
            ),
            RetrievalEvaluationCase(
                query="gym shirt",
                relevant_item_ids={"sku-gym-shirt-001"},
            ),
            RetrievalEvaluationCase(
                query="wireless headphones",
                relevant_item_ids={"sku-headphones-001"},
            ),
        ],
        recommendation_cases=[
            RecommendationEvaluationCase(
                user_id="user-runner",
                relevant_item_ids={"sku-trail-shoes-001"},
            ),
        ],
    )


def build_sample_benchmark_dataset() -> BenchmarkDataset:
    products = pl.DataFrame(
        {
            "item_id": [
                "shoe-road-runner",
                "shoe-trail-runner",
                "shoe-walking-comfort",
                "shirt-training-dry",
                "shorts-training-core",
                "hoodie-warmup-fleece",
                "headphones-wireless-pro",
                "earbuds-sport-fit",
                "speaker-bluetooth-mini",
                "yoga-mat-grip",
                "water-bottle-insulated",
                "fitness-tracker-pro",
            ],
            "title": [
                "Road Running Shoes",
                "Trail Running Shoes",
                "Comfort Walking Shoes",
                "Dry Training T-Shirt",
                "Core Training Shorts",
                "Warmup Fleece Hoodie",
                "Wireless Pro Headphones",
                "Sport Fit Earbuds",
                "Mini Bluetooth Speaker",
                "Grip Yoga Mat",
                "Insulated Water Bottle",
                "Pro Fitness Tracker",
            ],
            "category": [
                "Footwear",
                "Footwear",
                "Footwear",
                "Apparel",
                "Apparel",
                "Apparel",
                "Electronics",
                "Electronics",
                "Electronics",
                "Fitness",
                "Fitness",
                "Fitness",
            ],
            "brand": [
                "StrideLab",
                "StrideLab",
                "WalkWell",
                "CoreWear",
                "CoreWear",
                "CoreWear",
                "NorthAudio",
                "NorthAudio",
                "SoundPeak",
                "ZenFlex",
                "HydraGo",
                "PulseTrack",
            ],
            "description": [
                "Lightweight shoes for road running and daily training.",
                "Durable trail shoes with grip for outdoor running.",
                "Cushioned shoes for walking and all-day comfort.",
                "Breathable gym shirt for training and workouts.",
                "Flexible shorts for gym training and lifting sessions.",
                "Soft hoodie for warmups, recovery, and cool weather.",
                "Wireless headphones with noise isolation for music.",
                "Sweat-resistant wireless earbuds for workouts and runs.",
                "Portable Bluetooth speaker for music and travel.",
                "Non-slip yoga mat for stretching and floor workouts.",
                "Insulated bottle for water during training sessions.",
                "Fitness tracker for steps, heart rate, and workouts.",
            ],
            "price": [
                2999.0,
                3799.0,
                2499.0,
                899.0,
                1199.0,
                1999.0,
                4999.0,
                3499.0,
                1799.0,
                1299.0,
                799.0,
                5999.0,
            ],
        }
    )
    interactions = pl.DataFrame(
        {
            "user_id": [
                "user-runner",
                "user-runner",
                "user-audio",
                "user-audio",
                "user-yoga",
                "user-yoga",
                "user-gym",
                "user-gym",
            ],
            "item_id": [
                "shoe-road-runner",
                "shirt-training-dry",
                "headphones-wireless-pro",
                "speaker-bluetooth-mini",
                "yoga-mat-grip",
                "shirt-training-dry",
                "shirt-training-dry",
                "hoodie-warmup-fleece",
            ],
            "event_type": [
                "purchase",
                "click",
                "purchase",
                "click",
                "purchase",
                "view",
                "purchase",
                "click",
            ],
            "timestamp_ms": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    return BenchmarkDataset(
        name="sample",
        products=products,
        interactions=interactions,
        ranking_training_cases=[
            RetrievalEvaluationCase(
                query="road jogging shoes",
                relevant_item_ids={"shoe-road-runner"},
            ),
            RetrievalEvaluationCase(
                query="outdoor trail footwear",
                relevant_item_ids={"shoe-trail-runner"},
            ),
            RetrievalEvaluationCase(
                query="breathable workout shirt",
                relevant_item_ids={"shirt-training-dry"},
            ),
            RetrievalEvaluationCase(
                query="bluetooth personal audio",
                relevant_item_ids={"headphones-wireless-pro", "earbuds-sport-fit"},
            ),
            RetrievalEvaluationCase(
                query="non slip exercise mat",
                relevant_item_ids={"yoga-mat-grip"},
            ),
            RetrievalEvaluationCase(
                query="activity tracking watch",
                relevant_item_ids={"fitness-tracker-pro"},
            ),
        ],
        search_cases=[
            RetrievalEvaluationCase(
                query="running shoes",
                relevant_item_ids={"shoe-road-runner", "shoe-trail-runner"},
            ),
            RetrievalEvaluationCase(
                query="trail running shoes",
                relevant_item_ids={"shoe-trail-runner"},
            ),
            RetrievalEvaluationCase(
                query="gym training shirt",
                relevant_item_ids={"shirt-training-dry"},
            ),
            RetrievalEvaluationCase(
                query="wireless headphones",
                relevant_item_ids={"headphones-wireless-pro", "earbuds-sport-fit"},
            ),
            RetrievalEvaluationCase(
                query="yoga mat",
                relevant_item_ids={"yoga-mat-grip"},
            ),
            RetrievalEvaluationCase(
                query="fitness tracker",
                relevant_item_ids={"fitness-tracker-pro"},
            ),
        ],
        recommendation_cases=[
            RecommendationEvaluationCase(
                user_id="user-runner",
                relevant_item_ids={"shoe-trail-runner"},
            ),
            RecommendationEvaluationCase(
                user_id="user-audio",
                relevant_item_ids={"earbuds-sport-fit"},
            ),
            RecommendationEvaluationCase(
                user_id="user-yoga",
                relevant_item_ids={"water-bottle-insulated"},
            ),
            RecommendationEvaluationCase(
                user_id="user-gym",
                relevant_item_ids={"shorts-training-core"},
            ),
        ],
    )

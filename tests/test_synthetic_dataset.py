from pathlib import Path

import polars as pl

from commercemind.data.generate_synthetic_dataset import main
from commercemind.data.synthetic import (
    SyntheticDatasetSpec,
    build_synthetic_dataset,
    generate_synthetic_interactions,
    generate_synthetic_products,
    materialize_synthetic_dataset,
)


def test_synthetic_products_are_deterministic() -> None:
    first = generate_synthetic_products(10, seed=7)
    second = generate_synthetic_products(10, seed=7)

    assert first.equals(second)
    assert first.height == 10
    assert first.columns == [
        "item_id",
        "title",
        "category",
        "brand",
        "description",
        "price",
    ]


def test_synthetic_interactions_reference_generated_products() -> None:
    products = generate_synthetic_products(25, seed=7)
    interactions = generate_synthetic_interactions(
        products,
        user_count=5,
        interaction_count=50,
        seed=7,
    )

    product_ids = set(products["item_id"].to_list())
    interaction_item_ids = set(interactions["item_id"].to_list())

    assert interactions.height == 50
    assert interaction_item_ids <= product_ids
    assert interactions["user_id"].n_unique() <= 5


def test_build_synthetic_dataset_returns_products_and_interactions() -> None:
    products, interactions = build_synthetic_dataset(
        SyntheticDatasetSpec(
            dataset_name="synthetic_test",
            product_count=12,
            user_count=4,
            interaction_count=20,
            seed=11,
        )
    )

    assert products.height == 12
    assert interactions.height == 20


def test_materialize_synthetic_dataset_writes_processed_parquet(tmp_path: Path) -> None:
    result = materialize_synthetic_dataset(
        SyntheticDatasetSpec(
            dataset_name="synthetic_test",
            product_count=12,
            user_count=4,
            interaction_count=20,
            seed=11,
        ),
        processed_dir=tmp_path,
    )

    products = pl.read_parquet(result.products_path)
    interactions = pl.read_parquet(result.interactions_path)

    assert result.dataset_name == "synthetic_test"
    assert result.product_count == 12
    assert result.user_count == 4
    assert result.interaction_count == 20
    assert products.height == 12
    assert interactions.height == 20


def test_generate_synthetic_dataset_cli_writes_processed_files(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--dataset",
            "synthetic_cli",
            "--products",
            "10",
            "--users",
            "3",
            "--interactions",
            "15",
            "--seed",
            "5",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "products.parquet").exists()
    assert (tmp_path / "interactions.parquet").exists()

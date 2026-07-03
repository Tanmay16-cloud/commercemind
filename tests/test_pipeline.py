import polars as pl
import pytest

from commercemind.data.pipeline import materialize_normalized_dataset


def test_materialize_normalized_dataset_writes_parquet_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from commercemind.data.datasets import DatasetPaths

    dataset_name = "test_dataset"
    raw_dir = tmp_path / "raw" / dataset_name
    processed_dir = tmp_path / "processed" / dataset_name
    raw_dir.mkdir(parents=True)

    pl.DataFrame(
        {
            "parent_asin": ["A1"],
            "title": ["  Running Shoes  "],
            "main_category": ["Footwear"],
            "store": ["Nike"],
            "price": [2999.0],
        }
    ).write_ndjson(raw_dir / "products.jsonl")

    pl.DataFrame(
        {
            "user_id": ["U1"],
            "asin": ["A1"],
            "timestamp": [1720000000000],
            "rating": [4.0],
        }
    ).write_ndjson(raw_dir / "interactions.jsonl")

    monkeypatch.setattr(
        "commercemind.data.pipeline.get_dataset_paths",
        lambda name: DatasetPaths(dataset_name=name, raw_dir=raw_dir, processed_dir=processed_dir),
    )
    monkeypatch.setattr(
        "commercemind.data.ingest.get_dataset_paths",
        lambda name: DatasetPaths(dataset_name=name, raw_dir=raw_dir, processed_dir=processed_dir),
    )

    products, interactions = materialize_normalized_dataset(dataset_name)

    assert products.height == 1
    assert interactions.height == 1
    assert (processed_dir / "products.parquet").exists()
    assert (processed_dir / "interactions.parquet").exists()

from pathlib import Path

import pytest

from commercemind.data.datasets import DatasetPaths
from commercemind.data.ingest import IngestionError, load_raw_interactions, load_raw_products


def test_load_raw_products_raises_for_missing_file(tmp_path: Path) -> None:
    paths = DatasetPaths(
        dataset_name="test_dataset",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )

    with pytest.raises(IngestionError, match="products.jsonl"):
        load_raw_products(paths)


def test_load_raw_interactions_raises_for_missing_file(tmp_path: Path) -> None:
    paths = DatasetPaths(
        dataset_name="test_dataset",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )

    with pytest.raises(IngestionError, match="interactions.jsonl"):
        load_raw_interactions(paths)

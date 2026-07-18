from pathlib import Path

import polars as pl
import pytest

from commercemind.data.datasets import DatasetPaths
from commercemind.storage.database import build_engine, create_schema, create_session_factory
from commercemind.storage.load_dataset import (
    DatasetLoadError,
    load_processed_dataset,
    main,
    seed_database_from_processed_dataset,
)
from commercemind.storage.repositories import InteractionRepository, ProductRepository


def test_load_processed_dataset_reads_products_and_interactions(tmp_path: Path) -> None:
    paths = _write_processed_dataset(tmp_path)

    products, interactions = load_processed_dataset(paths)

    assert products["item_id"].to_list() == ["A1", "A2"]
    assert interactions["event_type"].to_list() == ["purchase"]


def test_load_processed_dataset_raises_for_missing_files(tmp_path: Path) -> None:
    paths = DatasetPaths(
        dataset_name="missing",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )

    with pytest.raises(DatasetLoadError, match="missing processed dataset file"):
        load_processed_dataset(paths)


def test_seed_database_from_processed_dataset_round_trips_rows(tmp_path: Path) -> None:
    paths = _write_processed_dataset(tmp_path)
    engine = build_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        result = seed_database_from_processed_dataset(session, paths)
        stored_products = ProductRepository(session).list_all()
        stored_interactions = InteractionRepository(session).list_all()

    assert result.dataset_name == "unit"
    assert result.product_count == 2
    assert result.interaction_count == 1
    assert stored_products["item_id"].to_list() == ["A1", "A2"]
    assert stored_interactions["user_id"].to_list() == ["U1"]


def test_load_dataset_cli_seeds_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _write_processed_dataset(tmp_path)
    db_path = tmp_path / "catalog.db"

    monkeypatch.setattr(
        "commercemind.storage.load_dataset.get_dataset_paths",
        lambda dataset_name: paths,
    )

    exit_code = main(
        [
            "--dataset",
            "unit",
            "--database-url",
            f"sqlite+pysqlite:///{db_path}",
        ]
    )

    assert exit_code == 0
    assert db_path.exists()


def _write_processed_dataset(tmp_path: Path) -> DatasetPaths:
    paths = DatasetPaths(
        dataset_name="unit",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
    )
    paths.processed_dir.mkdir(parents=True)

    pl.DataFrame(
        {
            "item_id": ["A1", "A2"],
            "title": ["Running Shoes", "Wireless Headphones"],
            "category": ["Footwear", "Electronics"],
            "brand": ["StrideLab", "NorthAudio"],
            "description": ["Road running shoe", "Bluetooth headphones"],
            "price": [2999.0, 2499.0],
        }
    ).write_parquet(paths.products_path)
    pl.DataFrame(
        {
            "user_id": ["U1"],
            "item_id": ["A1"],
            "event_type": ["purchase"],
            "timestamp_ms": [1],
        }
    ).write_parquet(paths.interactions_path)

    return paths

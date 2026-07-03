from pathlib import Path

import polars as pl

from commercemind.data.datasets import DatasetPaths, get_dataset_paths


class IngestionError(RuntimeError):
    pass


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise IngestionError(f"Required input file does not exist: {path}")
    return path


def load_raw_products(paths: DatasetPaths) -> pl.DataFrame:
    source_path = _require_file(paths.raw_dir / "products.jsonl")
    return pl.read_ndjson(source_path)


def load_raw_interactions(paths: DatasetPaths) -> pl.DataFrame:
    source_path = _require_file(paths.raw_dir / "interactions.jsonl")
    return pl.read_ndjson(source_path)


def build_dataset_frames(dataset_name: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    paths = get_dataset_paths(dataset_name)
    products = load_raw_products(paths)
    interactions = load_raw_interactions(paths)
    return products, interactions

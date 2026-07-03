import polars as pl

from commercemind.data.datasets import get_dataset_paths
from commercemind.data.ingest import build_dataset_frames
from commercemind.data.transform import normalize_interactions, normalize_products


def build_normalized_dataset(dataset_name: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    products, interactions = build_dataset_frames(dataset_name)

    normalized_products = normalize_products(products)
    normalized_interactions = normalize_interactions(interactions)

    return normalized_products, normalized_interactions


def materialize_normalized_dataset(dataset_name: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    paths = get_dataset_paths(dataset_name)
    products, interactions = build_normalized_dataset(dataset_name)

    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    products.write_parquet(paths.products_path)
    interactions.write_parquet(paths.interactions_path)

    return products, interactions

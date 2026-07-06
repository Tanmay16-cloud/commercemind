from dataclasses import dataclass
from pathlib import Path

from commercemind.config import get_settings

DEFAULT_DATASET_NAME = "amazon_fashion"


@dataclass(frozen=True)
class ProductRecord:
    item_id: str
    title: str
    category: str
    brand: str | None = None
    description: str | None = None
    price: float | None = None


@dataclass(frozen=True)
class InteractionRecord:
    user_id: str
    item_id: str
    event_type: str
    timestamp_ms: int
    rating: float | None = None


@dataclass(frozen=True)
class DatasetPaths:
    dataset_name: str
    raw_dir: Path
    processed_dir: Path

    @property
    def products_path(self) -> Path:
        return self.processed_dir / "products.parquet"

    @property
    def interactions_path(self) -> Path:
        return self.processed_dir / "interactions.parquet"


def get_dataset_paths(dataset_name: str = DEFAULT_DATASET_NAME) -> DatasetPaths:
    settings = get_settings()

    raw_dir = settings.raw_data_dir / dataset_name
    processed_dir = settings.processed_data_dir / dataset_name

    return DatasetPaths(
        dataset_name=dataset_name,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

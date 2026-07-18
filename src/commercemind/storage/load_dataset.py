from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
from sqlalchemy.orm import Session

from commercemind.data.datasets import DEFAULT_DATASET_NAME, DatasetPaths, get_dataset_paths
from commercemind.storage.database import build_engine, create_schema, create_session_factory
from commercemind.storage.seed import seed_catalog


class DatasetLoadError(Exception):
    pass


@dataclass(frozen=True)
class DatasetSeedResult:
    dataset_name: str
    product_count: int
    interaction_count: int


def load_processed_dataset(paths: DatasetPaths) -> tuple[pl.DataFrame, pl.DataFrame]:
    missing_paths = [
        path for path in [paths.products_path, paths.interactions_path] if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise DatasetLoadError(f"missing processed dataset file(s): {missing}")

    return pl.read_parquet(paths.products_path), pl.read_parquet(paths.interactions_path)


def seed_database_from_processed_dataset(
    session: Session,
    paths: DatasetPaths,
) -> DatasetSeedResult:
    products, interactions = load_processed_dataset(paths)
    seed_catalog(session, products=products, interactions=interactions)

    return DatasetSeedResult(
        dataset_name=paths.dataset_name,
        product_count=products.height,
        interaction_count=interactions.height,
    )


def seed_database_from_dataset(
    dataset_name: str = DEFAULT_DATASET_NAME,
    *,
    database_url: str | None = None,
    create_tables: bool = True,
) -> DatasetSeedResult:
    engine = build_engine(database_url)
    if create_tables:
        create_schema(engine)

    session_factory = create_session_factory(engine)
    with session_factory() as session:
        return seed_database_from_processed_dataset(session, get_dataset_paths(dataset_name))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed CommerceMind products and interactions from processed dataset files."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name under data/processed/.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy database URL. Defaults to COMMERCE_MIND_DATABASE_URL.",
    )
    parser.add_argument(
        "--skip-create-schema",
        action="store_true",
        help="Skip table creation before seeding.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = seed_database_from_dataset(
            args.dataset,
            database_url=args.database_url,
            create_tables=not args.skip_create_schema,
        )
    except DatasetLoadError as exc:
        parser.error(str(exc))

    print(
        "Seeded "
        f"{result.product_count} products and "
        f"{result.interaction_count} interactions "
        f"for dataset '{result.dataset_name}'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

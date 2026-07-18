from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from sqlalchemy.exc import SQLAlchemyError

from commercemind.services.sample_data import default_interactions, default_products
from commercemind.storage.database import build_engine, create_session_factory
from commercemind.storage.repositories import InteractionRepository, ProductRepository


@dataclass(frozen=True)
class CatalogData:
    products: pl.DataFrame
    interactions: pl.DataFrame


class CatalogLoadError(RuntimeError):
    pass


def load_sample_catalog() -> CatalogData:
    return CatalogData(
        products=default_products(),
        interactions=default_interactions(),
    )


def load_database_catalog(database_url: str | None = None) -> CatalogData:
    engine = build_engine(database_url)
    session_factory = create_session_factory(engine)

    try:
        with session_factory() as session:
            products = ProductRepository(session).list_all()
            interactions = InteractionRepository(session).list_all()
    except SQLAlchemyError as exc:
        raise CatalogLoadError("could not load catalog from database") from exc

    if products.is_empty():
        raise CatalogLoadError("database catalog has no products; seed it before starting the API")

    return CatalogData(products=products, interactions=interactions)


def load_catalog_data(source: str, database_url: str | None = None) -> CatalogData:
    normalized_source = source.strip().lower()

    if normalized_source == "sample":
        return load_sample_catalog()
    if normalized_source == "database":
        return load_database_catalog(database_url)

    raise CatalogLoadError(
        f"unknown catalog source '{source}'. Expected one of: sample, database"
    )

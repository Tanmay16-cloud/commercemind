from pathlib import Path

import pytest

from commercemind.services.catalog import (
    CatalogLoadError,
    load_catalog_data,
    load_database_catalog,
    load_sample_catalog,
)
from commercemind.storage.database import build_engine, create_schema, create_session_factory
from commercemind.storage.seed import seed_catalog


def test_sample_catalog_loads_default_products_and_interactions() -> None:
    catalog = load_sample_catalog()

    assert catalog.products["item_id"].to_list()[0] == "sku-running-shoes-001"
    assert catalog.interactions["user_id"].to_list()[0] == "user-runner"


def test_database_catalog_loads_seeded_products_and_interactions(tmp_path: Path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'catalog.db'}"
    engine = build_engine(database_url)
    create_schema(engine)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        seed_catalog(session)

    catalog = load_database_catalog(database_url)

    assert catalog.products["item_id"].to_list()[0] == "sku-gym-shirt-001"
    assert catalog.interactions["event_type"].to_list() == [
        "purchase",
        "click",
        "click",
        "purchase",
    ]


def test_database_catalog_rejects_empty_catalog(tmp_path: Path) -> None:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'empty.db'}"
    engine = build_engine(database_url)
    create_schema(engine)

    with pytest.raises(CatalogLoadError, match="no products"):
        load_database_catalog(database_url)


def test_catalog_source_must_be_known() -> None:
    with pytest.raises(CatalogLoadError, match="unknown catalog source"):
        load_catalog_data("spreadsheet")

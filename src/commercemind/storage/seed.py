from __future__ import annotations

import polars as pl
from sqlalchemy.orm import Session

from commercemind.services.sample_data import default_interactions, default_products
from commercemind.storage.models import InteractionRow, ProductRow
from commercemind.storage.repositories import InteractionRepository, ProductRepository


def seed_catalog(
    session: Session,
    products: pl.DataFrame | None = None,
    interactions: pl.DataFrame | None = None,
) -> None:
    session.query(InteractionRow).delete()
    session.query(ProductRow).delete()

    ProductRepository(session).replace_all(products if products is not None else default_products())
    InteractionRepository(session).replace_all(
        interactions if interactions is not None else default_interactions()
    )
    session.commit()

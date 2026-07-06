from __future__ import annotations

import polars as pl
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from commercemind.storage.models import InteractionRow, ProductRow


class ProductRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def replace_all(self, products: pl.DataFrame) -> None:
        self._session.execute(delete(ProductRow))
        self._session.add_all(
            [
                ProductRow(
                    item_id=str(row["item_id"]),
                    title=str(row["title"]),
                    category=_optional_text(row.get("category")),
                    brand=_optional_text(row.get("brand")),
                    description=_optional_text(row.get("description")),
                    price=_optional_float(row.get("price")),
                )
                for row in products.iter_rows(named=True)
            ]
        )

    def list_all(self) -> pl.DataFrame:
        rows = self._session.scalars(select(ProductRow).order_by(ProductRow.item_id)).all()
        return pl.DataFrame(
            {
                "item_id": [row.item_id for row in rows],
                "title": [row.title for row in rows],
                "category": [row.category for row in rows],
                "brand": [row.brand for row in rows],
                "description": [row.description for row in rows],
                "price": [row.price for row in rows],
            }
        )


class InteractionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def replace_all(self, interactions: pl.DataFrame) -> None:
        self._session.execute(delete(InteractionRow))
        self._session.add_all(
            [
                InteractionRow(
                    user_id=str(row["user_id"]),
                    item_id=str(row["item_id"]),
                    event_type=str(row["event_type"]),
                    timestamp_ms=int(row["timestamp_ms"]),
                )
                for row in interactions.iter_rows(named=True)
            ]
        )

    def list_all(self) -> pl.DataFrame:
        rows = self._session.scalars(
            select(InteractionRow).order_by(InteractionRow.timestamp_ms, InteractionRow.id)
        ).all()
        return pl.DataFrame(
            {
                "user_id": [row.user_id for row in rows],
                "item_id": [row.item_id for row in rows],
                "event_type": [row.event_type for row in rows],
                "timestamp_ms": [row.timestamp_ms for row in rows],
            }
        )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)

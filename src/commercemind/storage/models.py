from __future__ import annotations

from sqlalchemy import BigInteger, Float, ForeignKey, Index, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ProductRow(Base):
    __tablename__ = "products"
    __table_args__ = (
        Index("ix_products_category", "category"),
        Index("ix_products_brand", "brand"),
    )

    item_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[str | None] = mapped_column(String(128))
    brand: Mapped[str | None] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(String(4096))
    price: Mapped[float | None] = mapped_column(Float)


class InteractionRow(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        Index("ix_interactions_user_id", "user_id"),
        Index("ix_interactions_item_id", "item_id"),
        Index("ix_interactions_user_item", "user_id", "item_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    item_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("products.item_id"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)

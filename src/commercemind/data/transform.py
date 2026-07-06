import polars as pl


class TransformError(RuntimeError):
    pass


def _required_column(frame: pl.DataFrame, candidates: list[str], *, dtype: pl.DataType) -> pl.Expr:
    for name in candidates:
        if name in frame.columns:
            return pl.col(name).cast(dtype, strict=False)
    raise TransformError(f"Missing required column. Expected one of: {candidates}")


def _optional_column(frame: pl.DataFrame, candidates: list[str], *, dtype: pl.DataType) -> pl.Expr:
    for name in candidates:
        if name in frame.columns:
            return pl.col(name).cast(dtype, strict=False)
    return pl.lit(None, dtype=dtype)


def normalize_products(products: pl.DataFrame) -> pl.DataFrame:
    return products.select(
        _required_column(products, ["item_id", "parent_asin", "asin"], dtype=pl.Utf8).alias(
            "item_id"
        ),
        _required_column(products, ["title"], dtype=pl.Utf8).alias("title"),
        _optional_column(products, ["category", "main_category"], dtype=pl.Utf8).alias("category"),
        _optional_column(products, ["brand", "store"], dtype=pl.Utf8).alias("brand"),
        _optional_column(products, ["description"], dtype=pl.Utf8).alias("description"),
        _optional_column(products, ["price"], dtype=pl.Float64).alias("price"),
    ).with_columns(
        pl.col("title").str.strip_chars(),
        pl.col("category").fill_null("unknown"),
    )


def normalize_interactions(interactions: pl.DataFrame) -> pl.DataFrame:
    return interactions.select(
        _required_column(interactions, ["user_id"], dtype=pl.Utf8).alias("user_id"),
        _required_column(interactions, ["item_id", "parent_asin", "asin"], dtype=pl.Utf8).alias(
            "item_id"
        ),
        _optional_column(interactions, ["event_type"], dtype=pl.Utf8)
        .fill_null("rating")
        .alias("event_type"),
        _required_column(interactions, ["timestamp_ms", "timestamp"], dtype=pl.Int64).alias(
            "timestamp_ms"
        ),
        _optional_column(interactions, ["rating"], dtype=pl.Float64).alias("rating"),
    )

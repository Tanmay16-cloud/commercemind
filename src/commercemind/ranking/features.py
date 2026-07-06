from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from commercemind.retrieval.baseline import RetrievalCandidate, tokenize


@dataclass(frozen=True)
class ProductRankingFeatures:
    retrieval_score: float
    title_match: float
    category_match: float
    brand_match: float
    description_match: float
    price_intent_match: float


class ProductFeatureStore:
    def __init__(self, products: pl.DataFrame) -> None:
        self._products_by_item_id = {
            str(row["item_id"]): row for row in products.iter_rows(named=True)
        }
        prices = [
            float(row["price"])
            for row in self._products_by_item_id.values()
            if row.get("price") is not None
        ]
        self._min_price = min(prices, default=0.0)
        self._max_price = max(prices, default=0.0)

    def build_features(self,query: str,
    candidate: RetrievalCandidate,
    ) -> ProductRankingFeatures:
        row = self._products_by_item_id.get(candidate.item_id, {})
        query_tokens = set(tokenize(query))

        return ProductRankingFeatures(
            retrieval_score=candidate.score,
            title_match=_token_overlap(query_tokens, _row_text(row, "title")),
            category_match=_token_overlap(query_tokens, _row_text(row, "category")),
            brand_match=_token_overlap(query_tokens, _row_text(row, "brand")),
            description_match=_token_overlap(query_tokens, _row_text(row, "description")),
            price_intent_match=self._price_intent_match(query_tokens, row),
        )

    def _price_intent_match(self, query_tokens: set[str], row: dict[str, object]) -> float:
        price = row.get("price")
        if price is None:
            return 0.0

        normalized_price = _normalize_price(float(price), self._min_price, self._max_price)
        budget_terms = {"affordable", "budget", "cheap", "low", "value"}
        premium_terms = {"premium", "luxury", "expensive", "high", "pro"}

        if query_tokens & budget_terms:
            return 1.0 - normalized_price
        if query_tokens & premium_terms:
            return normalized_price
        return 0.0


def _row_text(row: dict[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    return str(value)


def _token_overlap(query_tokens: set[str], text: str | None) -> float:
    if not query_tokens:
        return 0.0

    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0

    return len(query_tokens & text_tokens) / len(query_tokens)


def _normalize_price(price: float, min_price: float, max_price: float) -> float:
    if max_price <= min_price:
        return 0.0
    return (price - min_price) / (max_price - min_price)

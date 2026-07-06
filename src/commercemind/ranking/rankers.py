from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import polars as pl

from commercemind.ranking.features import ProductFeatureStore, ProductRankingFeatures
from commercemind.retrieval.baseline import RetrievalCandidate


class ProductRanker(Protocol):
    def rank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        ...


@dataclass(frozen=True)
class RankingWeights:
    retrieval_score: float = 1.0
    title_match: float = 0.35
    category_match: float = 0.2
    brand_match: float = 0.1
    description_match: float = 0.15
    price_intent_match: float = 0.2


class LinearProductRanker:
    def __init__(
        self,
        products: pl.DataFrame,
        weights: RankingWeights | None = None,
    ) -> None:
        self._feature_store = ProductFeatureStore(products)
        self._weights = weights or RankingWeights()

    def rank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        if top_k <= 0:
            return []

        scored = [
            _ScoredCandidate(
                candidate=candidate,
                ranking_score=self._score(self._feature_store.build_features(query, candidate)),
            )
            for candidate in candidates
        ]
        ranked = sorted(
            scored,
            key=lambda item: (-item.ranking_score, -item.candidate.score, item.candidate.title),
        )

        return [
            RetrievalCandidate(
                item_id=item.candidate.item_id,
                title=item.candidate.title,
                score=item.ranking_score,
            )
            for item in ranked[:top_k]
        ]

    def _score(self, features: ProductRankingFeatures) -> float:
        return (
            self._weights.retrieval_score * features.retrieval_score
            + self._weights.title_match * features.title_match
            + self._weights.category_match * features.category_match
            + self._weights.brand_match * features.brand_match
            + self._weights.description_match * features.description_match
            + self._weights.price_intent_match * features.price_intent_match
        )


@dataclass(frozen=True)
class _ScoredCandidate:
    candidate: RetrievalCandidate
    ranking_score: float

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from commercemind.retrieval.baseline import RetrievalCandidate

EVENT_WEIGHTS = {
    "view": 1.0,
    "click": 2.0,
    "add_to_cart": 3.0,
    "purchase": 5.0,
}


@dataclass(frozen=True)
class UserPreferenceProfile:
    category_weights: dict[str, float]
    brand_weights: dict[str, float]
    seen_item_ids: set[str]

    @property
    def is_empty(self) -> bool:
        return not self.category_weights and not self.brand_weights

    def as_query(self) -> str:
        terms = [
            *sorted(self.category_weights, key=self.category_weights.__getitem__, reverse=True),
            *sorted(self.brand_weights, key=self.brand_weights.__getitem__, reverse=True),
        ]
        return " ".join(terms)


class PersonalizedCandidateGenerator:
    def __init__(self, products: pl.DataFrame, interactions: pl.DataFrame) -> None:
        self._products = [row for row in products.iter_rows(named=True)]
        self._products_by_item_id = {str(row["item_id"]): row for row in self._products}
        self._interactions = [row for row in interactions.iter_rows(named=True)]
        self._popularity = self._build_popularity_scores()

    def generate(self, user_id: str, top_k: int) -> list[RetrievalCandidate]:
        if top_k <= 0:
            return []

        profile = self.build_user_profile(user_id)
        if profile.is_empty:
            return self._popular_candidates(profile.seen_item_ids, top_k)

        scored: list[RetrievalCandidate] = []

        for row in self._products:
            item_id = str(row["item_id"])
            if item_id in profile.seen_item_ids:
                continue

            score = (
                self._preference_score(row, profile)
                + 0.2 * self._popularity.get(item_id, 0.0)
            )
            if score > 0:
                scored.append(
                    RetrievalCandidate(
                        item_id=item_id,
                        title=str(row["title"]),
                        score=score,
                    )
                )

        ranked = sorted(scored, key=lambda candidate: (-candidate.score, candidate.title))
        if len(ranked) >= top_k:
            return ranked[:top_k]

        ranked_item_ids = {candidate.item_id for candidate in ranked}
        fallback = [
            candidate
            for candidate in self._popular_candidates(profile.seen_item_ids, top_k)
            if candidate.item_id not in ranked_item_ids
        ]
        return [*ranked, *fallback][:top_k]

    def build_user_profile(self, user_id: str) -> UserPreferenceProfile:
        category_weights: dict[str, float] = {}
        brand_weights: dict[str, float] = {}
        seen_item_ids: set[str] = set()

        for interaction in self._interactions:
            if str(interaction["user_id"]) != user_id:
                continue

            item_id = str(interaction["item_id"])
            product = self._products_by_item_id.get(item_id)
            if product is None:
                continue

            seen_item_ids.add(item_id)
            weight = EVENT_WEIGHTS.get(str(interaction.get("event_type", "view")), 1.0)
            _add_weight(category_weights, product.get("category"), weight)
            _add_weight(brand_weights, product.get("brand"), weight)

        return UserPreferenceProfile(
            category_weights=dict(category_weights),
            brand_weights=dict(brand_weights),
            seen_item_ids=seen_item_ids,
        )

    def _popular_candidates(
        self,
        excluded_item_ids: set[str],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        candidates = [
            RetrievalCandidate(
                item_id=str(row["item_id"]),
                title=str(row["title"]),
                score=self._popularity.get(str(row["item_id"]), 0.0),
            )
            for row in self._products
            if str(row["item_id"]) not in excluded_item_ids
        ]
        ranked = sorted(candidates, key=lambda candidate: (-candidate.score, candidate.title))
        return ranked[:top_k]

    def _preference_score(
        self,
        row: dict[str, object],
        profile: UserPreferenceProfile,
    ) -> float:
        category_score = profile.category_weights.get(str(row.get("category")), 0.0)
        brand_score = profile.brand_weights.get(str(row.get("brand")), 0.0)
        return category_score + 0.5 * brand_score

    def _build_popularity_scores(self) -> dict[str, float]:
        scores: dict[str, float] = {}

        for interaction in self._interactions:
            item_id = str(interaction["item_id"])
            weight = EVENT_WEIGHTS.get(str(interaction.get("event_type", "view")), 1.0)
            scores[item_id] = scores.get(item_id, 0.0) + weight

        max_score = max(scores.values(), default=0.0)
        if max_score == 0:
            return {}

        return {item_id: score / max_score for item_id, score in scores.items()}


def _add_weight(counter: dict[str, float], value: object, weight: float) -> None:
    if value is not None:
        key = str(value)
        counter[key] = counter.get(key, 0.0) + weight

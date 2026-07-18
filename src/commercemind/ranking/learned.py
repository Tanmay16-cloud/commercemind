from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from commercemind.ranking.features import (
    RANKING_FEATURE_NAMES,
    ProductFeatureStore,
    ProductRankingFeatures,
)
from commercemind.retrieval.baseline import RetrievalCandidate

RANKER_ARTIFACT_VERSION = 1


class LearnedRankerError(RuntimeError):
    pass


@dataclass(frozen=True)
class LearnedRankingModel:
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float
    training_examples: int
    positive_examples: int

    def __post_init__(self) -> None:
        if self.feature_names != RANKING_FEATURE_NAMES:
            raise LearnedRankerError("model feature names do not match ranking features")
        if len(self.weights) != len(self.feature_names):
            raise LearnedRankerError("model weights do not match feature count")


class LearnedLinearProductRanker:
    def __init__(self, products: pl.DataFrame, model: LearnedRankingModel) -> None:
        self._feature_store = ProductFeatureStore(products)
        self._model = model

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
                ranking_score=score_features(
                    self._feature_store.build_features(query, candidate),
                    self._model,
                ),
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

    @classmethod
    def load(cls, products: pl.DataFrame, path: Path) -> LearnedLinearProductRanker:
        return cls(products=products, model=load_learned_ranking_model(path))


def score_features(features: ProductRankingFeatures, model: LearnedRankingModel) -> float:
    return sum(
        value * weight
        for value, weight in zip(features.as_vector(), model.weights, strict=True)
    ) + model.bias


def save_learned_ranking_model(model: LearnedRankingModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": RANKER_ARTIFACT_VERSION,
        "model": {
            **asdict(model),
            "feature_names": list(model.feature_names),
            "weights": list(model.weights),
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_learned_ranking_model(path: Path) -> LearnedRankingModel:
    if not path.exists():
        raise LearnedRankerError(f"missing learned ranker artifact: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != RANKER_ARTIFACT_VERSION:
        raise LearnedRankerError("unsupported learned ranker artifact version")

    model_payload = payload["model"]
    return LearnedRankingModel(
        feature_names=tuple(model_payload["feature_names"]),
        weights=tuple(float(weight) for weight in model_payload["weights"]),
        bias=float(model_payload["bias"]),
        training_examples=int(model_payload["training_examples"]),
        positive_examples=int(model_payload["positive_examples"]),
    )


@dataclass(frozen=True)
class _ScoredCandidate:
    candidate: RetrievalCandidate
    ranking_score: float

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from commercemind.retrieval.metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from commercemind.schemas import RecommendationRequest, RecommendationResponse


class Recommender(Protocol):
    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        ...


@dataclass(frozen=True)
class RecommendationEvaluationCase:
    user_id: str
    relevant_item_ids: set[str]


@dataclass(frozen=True)
class RecommendationCaseMetrics:
    user_id: str
    recommended_item_ids: list[str]
    relevant_item_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    reciprocal_rank_at_k: float


@dataclass(frozen=True)
class RecommendationEvaluationSummary:
    case_count: int
    k: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_hit_rate_at_k: float
    mean_reciprocal_rank_at_k: float


@dataclass(frozen=True)
class RecommendationEvaluationReport:
    summary: RecommendationEvaluationSummary
    cases: list[RecommendationCaseMetrics]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_recommender(
    recommender: Recommender,
    cases: list[RecommendationEvaluationCase],
    *,
    k: int,
) -> RecommendationEvaluationReport:
    if k <= 0:
        raise ValueError("k must be positive")
    if not cases:
        raise ValueError("at least one evaluation case is required")

    case_metrics = [_evaluate_case(recommender, case, k) for case in cases]

    summary = RecommendationEvaluationSummary(
        case_count=len(case_metrics),
        k=k,
        mean_precision_at_k=_mean(metric.precision_at_k for metric in case_metrics),
        mean_recall_at_k=_mean(metric.recall_at_k for metric in case_metrics),
        mean_hit_rate_at_k=_mean(metric.hit_rate_at_k for metric in case_metrics),
        mean_reciprocal_rank_at_k=_mean(
            metric.reciprocal_rank_at_k for metric in case_metrics
        ),
    )

    return RecommendationEvaluationReport(summary=summary, cases=case_metrics)


def write_recommendation_report(report: RecommendationEvaluationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def _evaluate_case(
    recommender: Recommender,
    case: RecommendationEvaluationCase,
    k: int,
) -> RecommendationCaseMetrics:
    response = recommender.recommend(RecommendationRequest(user_id=case.user_id, top_k=k))
    recommended_item_ids = [item.item_id for item in response.results]

    return RecommendationCaseMetrics(
        user_id=case.user_id,
        recommended_item_ids=recommended_item_ids,
        relevant_item_ids=sorted(case.relevant_item_ids),
        precision_at_k=precision_at_k(case.relevant_item_ids, recommended_item_ids, k),
        recall_at_k=recall_at_k(case.relevant_item_ids, recommended_item_ids, k),
        hit_rate_at_k=hit_rate_at_k(case.relevant_item_ids, recommended_item_ids, k),
        reciprocal_rank_at_k=reciprocal_rank_at_k(
            case.relevant_item_ids,
            recommended_item_ids,
            k,
        ),
    )


def _mean(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers)

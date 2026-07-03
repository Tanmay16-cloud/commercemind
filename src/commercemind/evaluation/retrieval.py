from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

from commercemind.retrieval.baseline import RetrievalCandidate
from commercemind.retrieval.metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


class SearchRetriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        ...


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    query: str
    relevant_item_ids: set[str]


@dataclass(frozen=True)
class RetrievalCaseMetrics:
    query: str
    retrieved_item_ids: list[str]
    relevant_item_ids: list[str]
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    reciprocal_rank_at_k: float


@dataclass(frozen=True)
class RetrievalEvaluationSummary:
    case_count: int
    k: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_hit_rate_at_k: float
    mean_reciprocal_rank_at_k: float


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    summary: RetrievalEvaluationSummary
    cases: list[RetrievalCaseMetrics]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_search_retriever(
    retriever: SearchRetriever,
    cases: list[RetrievalEvaluationCase],
    *,
    k: int,
) -> RetrievalEvaluationReport:
    if k <= 0:
        raise ValueError("k must be positive")
    if not cases:
        raise ValueError("at least one evaluation case is required")

    case_metrics = [_evaluate_case(retriever, case, k) for case in cases]

    summary = RetrievalEvaluationSummary(
        case_count=len(case_metrics),
        k=k,
        mean_precision_at_k=_mean(metric.precision_at_k for metric in case_metrics),
        mean_recall_at_k=_mean(metric.recall_at_k for metric in case_metrics),
        mean_hit_rate_at_k=_mean(metric.hit_rate_at_k for metric in case_metrics),
        mean_reciprocal_rank_at_k=_mean(
            metric.reciprocal_rank_at_k for metric in case_metrics
        ),
    )

    return RetrievalEvaluationReport(summary=summary, cases=case_metrics)


def write_retrieval_report(report: RetrievalEvaluationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def _evaluate_case(
    retriever: SearchRetriever,
    case: RetrievalEvaluationCase,
    k: int,
) -> RetrievalCaseMetrics:
    candidates = retriever.retrieve(case.query, top_k=k)
    retrieved_item_ids = [candidate.item_id for candidate in candidates]

    return RetrievalCaseMetrics(
        query=case.query,
        retrieved_item_ids=retrieved_item_ids,
        relevant_item_ids=sorted(case.relevant_item_ids),
        precision_at_k=precision_at_k(case.relevant_item_ids, retrieved_item_ids, k),
        recall_at_k=recall_at_k(case.relevant_item_ids, retrieved_item_ids, k),
        hit_rate_at_k=hit_rate_at_k(case.relevant_item_ids, retrieved_item_ids, k),
        reciprocal_rank_at_k=reciprocal_rank_at_k(
            case.relevant_item_ids,
            retrieved_item_ids,
            k,
        ),
    )


def _mean(values: Iterable[float]) -> float:
    numbers = list(values)
    return sum(numbers) / len(numbers)

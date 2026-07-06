from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from commercemind.evaluation.recommendation import (
    RecommendationEvaluationCase,
    RecommendationEvaluationReport,
    evaluate_recommender,
)
from commercemind.evaluation.retrieval import (
    RetrievalEvaluationCase,
    RetrievalEvaluationReport,
    evaluate_search_retriever,
)
from commercemind.retrieval.baseline import LexicalRetriever, RetrievalCandidate
from commercemind.retrieval.hybrid import HybridRetriever
from commercemind.retrieval.vector import VectorRetriever
from commercemind.schemas import SearchRequest
from commercemind.services.recommendations import RecommendationService
from commercemind.services.sample_data import default_interactions, default_products
from commercemind.services.search import SearchService


@dataclass(frozen=True)
class ExperimentComparisonRow:
    task: str
    system_name: str
    precision_at_k: float
    recall_at_k: float
    hit_rate_at_k: float
    reciprocal_rank_at_k: float


@dataclass(frozen=True)
class OfflineExperimentReport:
    k: int
    search_reports: dict[str, RetrievalEvaluationReport]
    recommendation_reports: dict[str, RecommendationEvaluationReport]

    def comparison_rows(self) -> list[ExperimentComparisonRow]:
        rows: list[ExperimentComparisonRow] = []

        for system_name, report in self.search_reports.items():
            rows.append(
                ExperimentComparisonRow(
                    task="search",
                    system_name=system_name,
                    precision_at_k=report.summary.mean_precision_at_k,
                    recall_at_k=report.summary.mean_recall_at_k,
                    hit_rate_at_k=report.summary.mean_hit_rate_at_k,
                    reciprocal_rank_at_k=report.summary.mean_reciprocal_rank_at_k,
                )
            )

        for system_name, report in self.recommendation_reports.items():
            rows.append(
                ExperimentComparisonRow(
                    task="recommendation",
                    system_name=system_name,
                    precision_at_k=report.summary.mean_precision_at_k,
                    recall_at_k=report.summary.mean_recall_at_k,
                    hit_rate_at_k=report.summary.mean_hit_rate_at_k,
                    reciprocal_rank_at_k=report.summary.mean_reciprocal_rank_at_k,
                )
            )

        return rows

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "comparison": [row.__dict__ for row in self.comparison_rows()],
            "search_reports": {
                name: report.to_dict() for name, report in self.search_reports.items()
            },
            "recommendation_reports": {
                name: report.to_dict()
                for name, report in self.recommendation_reports.items()
            },
        }


def default_search_evaluation_cases() -> list[RetrievalEvaluationCase]:
    return [
        RetrievalEvaluationCase(
            query="running shoes",
            relevant_item_ids={"sku-running-shoes-001", "sku-trail-shoes-001"},
        ),
        RetrievalEvaluationCase(
            query="gym shirt",
            relevant_item_ids={"sku-gym-shirt-001"},
        ),
        RetrievalEvaluationCase(
            query="wireless headphones",
            relevant_item_ids={"sku-headphones-001"},
        ),
    ]


def default_recommendation_evaluation_cases() -> list[RecommendationEvaluationCase]:
    return [
        RecommendationEvaluationCase(
            user_id="user-runner",
            relevant_item_ids={"sku-trail-shoes-001"},
        ),
    ]


def run_default_offline_experiment(
    *,
    products: pl.DataFrame | None = None,
    interactions: pl.DataFrame | None = None,
    k: int = 2,
) -> OfflineExperimentReport:
    if k <= 0:
        raise ValueError("k must be positive")

    catalog = products if products is not None else default_products()
    events = interactions if interactions is not None else default_interactions()

    lexical = LexicalRetriever(catalog)
    vector = VectorRetriever(catalog)
    hybrid = HybridRetriever([LexicalRetriever(catalog), VectorRetriever(catalog)])
    ranked_search = _SearchServiceAdapter(SearchService(products=catalog))

    search_cases = default_search_evaluation_cases()
    recommendation_cases = default_recommendation_evaluation_cases()

    return OfflineExperimentReport(
        k=k,
        search_reports={
            "lexical": evaluate_search_retriever(lexical, search_cases, k=k),
            "vector": evaluate_search_retriever(vector, search_cases, k=k),
            "hybrid": evaluate_search_retriever(hybrid, search_cases, k=k),
            "hybrid_ranked": evaluate_search_retriever(ranked_search, search_cases, k=k),
        },
        recommendation_reports={
            "personalized": evaluate_recommender(
                RecommendationService(products=catalog, interactions=events),
                recommendation_cases,
                k=k,
            )
        },
    )


def write_experiment_json_report(report: OfflineExperimentReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def write_experiment_markdown_report(report: OfflineExperimentReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "| Task | System | Precision@K | Recall@K | HitRate@K | MRR@K |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    rows.extend(_markdown_row(row) for row in report.comparison_rows())
    path.write_text(
        "\n".join(["# CommerceMind Offline Evaluation", "", *rows, ""]),
        encoding="utf-8",
    )


class _SearchServiceAdapter:
    def __init__(self, service: SearchService) -> None:
        self._service = service

    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        response = self._service.search(SearchRequest(query=query, top_k=top_k))
        return [
            RetrievalCandidate(item_id=item.item_id, title=item.title, score=item.score)
            for item in response.results
        ]


def _markdown_row(row: ExperimentComparisonRow) -> str:
    return (
        f"| {row.task} | {row.system_name} | "
        f"{row.precision_at_k:.3f} | "
        f"{row.recall_at_k:.3f} | "
        f"{row.hit_rate_at_k:.3f} | "
        f"{row.reciprocal_rank_at_k:.3f} |"
    )

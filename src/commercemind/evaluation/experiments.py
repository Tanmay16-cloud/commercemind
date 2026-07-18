from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from commercemind.data.datasets import DatasetPaths, get_dataset_paths
from commercemind.evaluation.benchmarks import build_benchmark_dataset
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
from commercemind.ranking.learned import LearnedLinearProductRanker
from commercemind.ranking.training import (
    build_ranking_training_examples,
    train_learned_ranking_model,
)
from commercemind.retrieval.baseline import LexicalRetriever, RetrievalCandidate
from commercemind.retrieval.hybrid import HybridRetriever
from commercemind.retrieval.vector import VectorRetriever
from commercemind.schemas import SearchRequest
from commercemind.services.recommendations import RecommendationService
from commercemind.services.search import SearchService
from commercemind.storage.load_dataset import load_processed_dataset


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
    benchmark_name: str
    k: int
    search_reports: dict[str, RetrievalEvaluationReport]
    recommendation_reports: dict[str, RecommendationEvaluationReport]
    metadata: dict[str, object] | None = None

    def comparison_rows(self) -> list[ExperimentComparisonRow]:
        rows: list[ExperimentComparisonRow] = []

        for system_name, search_report in self.search_reports.items():
            rows.append(
                ExperimentComparisonRow(
                    task="search",
                    system_name=system_name,
                    precision_at_k=search_report.summary.mean_precision_at_k,
                    recall_at_k=search_report.summary.mean_recall_at_k,
                    hit_rate_at_k=search_report.summary.mean_hit_rate_at_k,
                    reciprocal_rank_at_k=search_report.summary.mean_reciprocal_rank_at_k,
                )
            )

        for system_name, recommendation_report in self.recommendation_reports.items():
            rows.append(
                ExperimentComparisonRow(
                    task="recommendation",
                    system_name=system_name,
                    precision_at_k=recommendation_report.summary.mean_precision_at_k,
                    recall_at_k=recommendation_report.summary.mean_recall_at_k,
                    hit_rate_at_k=recommendation_report.summary.mean_hit_rate_at_k,
                    reciprocal_rank_at_k=(
                        recommendation_report.summary.mean_reciprocal_rank_at_k
                    ),
                )
            )

        return rows

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_name": self.benchmark_name,
            "k": self.k,
            "metadata": self.metadata or {},
            "comparison": [row.__dict__ for row in self.comparison_rows()],
            "search_reports": {
                name: report.to_dict() for name, report in self.search_reports.items()
            },
            "recommendation_reports": {
                name: report.to_dict()
                for name, report in self.recommendation_reports.items()
            },
        }


@dataclass(frozen=True)
class RecommendationHoldoutSplit:
    training_interactions: pl.DataFrame
    cases: list[RecommendationEvaluationCase]
    skipped_user_count: int


def default_search_evaluation_cases() -> list[RetrievalEvaluationCase]:
    return build_benchmark_dataset("sample").search_cases


def default_recommendation_evaluation_cases() -> list[RecommendationEvaluationCase]:
    return build_benchmark_dataset("sample").recommendation_cases


def run_default_offline_experiment(
    *,
    products: pl.DataFrame | None = None,
    interactions: pl.DataFrame | None = None,
    benchmark_name: str = "sample",
    k: int = 2,
) -> OfflineExperimentReport:
    if k <= 0:
        raise ValueError("k must be positive")

    benchmark = build_benchmark_dataset(benchmark_name)
    catalog = products if products is not None else benchmark.products
    events = interactions if interactions is not None else benchmark.interactions

    lexical = LexicalRetriever(catalog)
    vector = VectorRetriever(catalog)
    hybrid = HybridRetriever([LexicalRetriever(catalog), VectorRetriever(catalog)])
    ranked_search = _SearchServiceAdapter(SearchService(products=catalog))
    learned_model = train_learned_ranking_model(
        build_ranking_training_examples(catalog, benchmark.ranking_training_cases)
    )
    learned_search = _SearchServiceAdapter(
        SearchService(
            products=catalog,
            ranker=LearnedLinearProductRanker(catalog, learned_model),
        )
    )

    return OfflineExperimentReport(
        benchmark_name=benchmark.name,
        k=k,
        search_reports={
            "lexical": evaluate_search_retriever(lexical, benchmark.search_cases, k=k),
            "vector": evaluate_search_retriever(vector, benchmark.search_cases, k=k),
            "hybrid": evaluate_search_retriever(hybrid, benchmark.search_cases, k=k),
            "hybrid_ranked": evaluate_search_retriever(
                ranked_search,
                benchmark.search_cases,
                k=k,
            ),
            "learned_ranked": evaluate_search_retriever(
                learned_search,
                benchmark.search_cases,
                k=k,
            ),
        },
        recommendation_reports={
            "personalized": evaluate_recommender(
                RecommendationService(products=catalog, interactions=events),
                benchmark.recommendation_cases,
                k=k,
            )
        },
    )


def run_processed_recommendation_experiment(
    dataset_name: str,
    *,
    dataset_paths: DatasetPaths | None = None,
    k: int = 5,
) -> OfflineExperimentReport:
    if k <= 0:
        raise ValueError("k must be positive")

    paths = dataset_paths or get_dataset_paths(dataset_name)
    products, interactions = load_processed_dataset(paths)
    split = build_recommendation_holdout_split(interactions)
    report = evaluate_recommender(
        RecommendationService(products=products, interactions=split.training_interactions),
        split.cases,
        k=k,
    )

    return OfflineExperimentReport(
        benchmark_name=paths.dataset_name,
        k=k,
        search_reports={},
        recommendation_reports={"personalized_holdout": report},
        metadata={
            "evaluation_type": "processed_recommendation_holdout",
            "product_count": products.height,
            "interaction_count": interactions.height,
            "training_interaction_count": split.training_interactions.height,
            "evaluated_user_count": len(split.cases),
            "skipped_user_count": split.skipped_user_count,
        },
    )


def build_recommendation_holdout_split(
    interactions: pl.DataFrame,
    *,
    min_history: int = 1,
    holdout_fraction: float = 0.2,
) -> RecommendationHoldoutSplit:
    if min_history <= 0:
        raise ValueError("min_history must be positive")
    if holdout_fraction <= 0 or holdout_fraction >= 1:
        raise ValueError("holdout_fraction must be between 0 and 1")

    required_columns = {"user_id", "item_id", "timestamp_ms"}
    missing_columns = required_columns - set(interactions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"interactions missing required column(s): {missing}")

    users: dict[str, list[dict[str, object]]] = {}
    sorted_interactions = interactions.sort(["user_id", "timestamp_ms"])
    for row in sorted_interactions.iter_rows(named=True):
        users.setdefault(str(row["user_id"]), []).append(row)

    training_rows: list[dict[str, object]] = []
    cases: list[RecommendationEvaluationCase] = []

    for user_id, rows in users.items():
        split_index = _holdout_split_index(rows, min_history, holdout_fraction)
        if split_index is None:
            continue

        history_rows = rows[:split_index]
        history_item_ids = {str(row["item_id"]) for row in history_rows}
        relevant_item_ids = {
            str(row["item_id"])
            for row in rows[split_index:]
            if str(row["item_id"]) not in history_item_ids
        }
        if not relevant_item_ids:
            continue

        training_rows.extend(history_rows)
        cases.append(
            RecommendationEvaluationCase(
                user_id=user_id,
                relevant_item_ids=relevant_item_ids,
            )
        )

    if not cases:
        raise ValueError("not enough user histories to build recommendation holdout cases")

    return RecommendationHoldoutSplit(
        training_interactions=pl.DataFrame(training_rows),
        cases=cases,
        skipped_user_count=len(users) - len(cases),
    )


def write_experiment_json_report(report: OfflineExperimentReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def write_experiment_markdown_report(report: OfflineExperimentReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    if report.metadata:
        metadata_rows = [
            "## Metadata",
            "",
            *[
                f"- `{key}`: `{value}`"
                for key, value in sorted(report.metadata.items())
            ],
            "",
        ]

    rows = [
        "| Task | System | Precision@K | Recall@K | HitRate@K | MRR@K |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    rows.extend(_markdown_row(row) for row in report.comparison_rows())
    path.write_text(
        "\n".join(
            [
                "# CommerceMind Offline Evaluation",
                "",
                f"Benchmark: `{report.benchmark_name}`",
                "",
                *metadata_rows,
                *rows,
                "",
            ]
        ),
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


def _holdout_split_index(
    rows: list[dict[str, object]],
    min_history: int,
    holdout_fraction: float,
) -> int | None:
    if len(rows) <= min_history:
        return None

    split_index = int(len(rows) * (1 - holdout_fraction))
    split_index = max(min_history, split_index)
    split_index = min(split_index, len(rows) - 1)
    return split_index


def _markdown_row(row: ExperimentComparisonRow) -> str:
    return (
        f"| {row.task} | {row.system_name} | "
        f"{row.precision_at_k:.3f} | "
        f"{row.recall_at_k:.3f} | "
        f"{row.hit_rate_at_k:.3f} | "
        f"{row.reciprocal_rank_at_k:.3f} |"
    )

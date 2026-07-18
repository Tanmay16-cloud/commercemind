from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from commercemind.config import get_settings
from commercemind.evaluation.benchmarks import build_benchmark_dataset
from commercemind.evaluation.retrieval import RetrievalEvaluationCase
from commercemind.ranking.features import (
    RANKING_FEATURE_NAMES,
    ProductFeatureStore,
    ProductRankingFeatures,
)
from commercemind.ranking.learned import LearnedRankingModel, save_learned_ranking_model
from commercemind.retrieval.baseline import LexicalRetriever, RetrievalCandidate
from commercemind.retrieval.hybrid import HybridRetriever, SearchRetriever
from commercemind.retrieval.vector import VectorRetriever


@dataclass(frozen=True)
class RankingTrainingExample:
    query: str
    item_id: str
    title: str
    label: float
    features: ProductRankingFeatures


@dataclass(frozen=True)
class RankerTrainingResult:
    benchmark_name: str
    model_path: Path
    training_examples: int
    positive_examples: int
    feature_count: int


def build_ranking_training_examples(
    products: pl.DataFrame,
    cases: Sequence[RetrievalEvaluationCase],
    *,
    retriever: SearchRetriever | None = None,
    candidate_pool_size: int = 20,
) -> list[RankingTrainingExample]:
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive")

    feature_store = ProductFeatureStore(products)
    search_retriever = retriever or HybridRetriever(
        [LexicalRetriever(products), VectorRetriever(products)]
    )
    products_by_item_id = {
        str(row["item_id"]): row for row in products.iter_rows(named=True)
    }
    examples: list[RankingTrainingExample] = []

    for case in cases:
        candidate_by_item_id = {
            candidate.item_id: candidate
            for candidate in search_retriever.retrieve(case.query, candidate_pool_size)
        }

        for item_id, row in products_by_item_id.items():
            candidate = candidate_by_item_id.get(
                item_id,
                RetrievalCandidate(
                    item_id=item_id,
                    title=str(row["title"]),
                    score=0.0,
                ),
            )
            examples.append(
                RankingTrainingExample(
                    query=case.query,
                    item_id=item_id,
                    title=candidate.title,
                    label=1.0 if item_id in case.relevant_item_ids else 0.0,
                    features=feature_store.build_features(case.query, candidate),
                )
            )

    return examples


def train_learned_ranking_model(
    examples: Sequence[RankingTrainingExample],
    *,
    l2_regularization: float = 1.0,
) -> LearnedRankingModel:
    if not examples:
        raise ValueError("at least one training example is required")
    if l2_regularization < 0:
        raise ValueError("l2_regularization must be non-negative")

    positive_examples = sum(1 for example in examples if example.label > 0)
    if positive_examples == 0:
        raise ValueError("at least one positive training example is required")
    if positive_examples == len(examples):
        raise ValueError("at least one negative training example is required")

    x = np.array([example.features.as_vector() for example in examples], dtype=np.float64)
    y = np.array([example.label for example in examples], dtype=np.float64)
    x_with_bias = np.column_stack([np.ones(len(examples)), x])
    penalty = np.eye(x_with_bias.shape[1], dtype=np.float64) * l2_regularization
    penalty[0, 0] = 0.0

    normal_matrix = x_with_bias.T @ x_with_bias + penalty
    target_vector = x_with_bias.T @ y

    try:
        parameters = np.linalg.solve(normal_matrix, target_vector)
    except np.linalg.LinAlgError:
        parameters = np.linalg.pinv(normal_matrix) @ target_vector

    return LearnedRankingModel(
        feature_names=RANKING_FEATURE_NAMES,
        weights=tuple(float(value) for value in parameters[1:]),
        bias=float(parameters[0]),
        training_examples=len(examples),
        positive_examples=positive_examples,
    )


def train_ranker_from_benchmark(
    benchmark_name: str = "sample",
    *,
    output_path: Path | None = None,
    candidate_pool_size: int = 20,
    l2_regularization: float = 1.0,
) -> RankerTrainingResult:
    benchmark = build_benchmark_dataset(benchmark_name)
    examples = build_ranking_training_examples(
        benchmark.products,
        benchmark.ranking_training_cases,
        candidate_pool_size=candidate_pool_size,
    )
    model = train_learned_ranking_model(
        examples,
        l2_regularization=l2_regularization,
    )
    model_path = output_path or _default_model_path()
    save_learned_ranking_model(model, model_path)

    return RankerTrainingResult(
        benchmark_name=benchmark.name,
        model_path=model_path,
        training_examples=model.training_examples,
        positive_examples=model.positive_examples,
        feature_count=len(model.feature_names),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a lightweight learned ranker from labeled benchmark queries."
    )
    parser.add_argument(
        "--benchmark",
        choices=["demo", "sample"],
        default="sample",
        help="Named benchmark dataset used for ranker training.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path for the learned ranker JSON artifact.",
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=int,
        default=20,
        help="Number of retrieved candidates used before full-catalog negative backfill.",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=1.0,
        help="L2 regularization strength for the linear model.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = train_ranker_from_benchmark(
            args.benchmark,
            output_path=args.output_path,
            candidate_pool_size=args.candidate_pool_size,
            l2_regularization=args.l2,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(
        "Trained learned ranker "
        f"from '{result.benchmark_name}' benchmark with "
        f"{result.training_examples} examples, "
        f"{result.positive_examples} positives, "
        f"{result.feature_count} features, "
        f"at {result.model_path}."
    )
    return 0


def _default_model_path() -> Path:
    settings = get_settings()
    if settings.models_dir is None:
        raise ValueError("models_dir is not configured")
    return settings.models_dir / "ranker.json"


if __name__ == "__main__":
    raise SystemExit(main())

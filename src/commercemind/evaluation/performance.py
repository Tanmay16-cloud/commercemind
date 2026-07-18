from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import polars as pl

from commercemind.data.datasets import DatasetPaths, get_dataset_paths
from commercemind.evaluation.benchmarks import build_benchmark_dataset
from commercemind.schemas import RecommendationRequest, SearchRequest
from commercemind.services.recommendations import RecommendationService
from commercemind.services.search import SearchService
from commercemind.storage.load_dataset import load_processed_dataset


@dataclass(frozen=True)
class RequestMeasurement:
    task: str
    latency_ms: float
    result_count: int
    failed: bool


@dataclass(frozen=True)
class PerformanceSummary:
    task: str
    request_count: int
    error_count: int
    empty_result_count: int
    total_latency_ms: float
    average_latency_ms: float
    min_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    throughput_requests_per_second: float
    error_rate: float
    empty_result_rate: float


@dataclass(frozen=True)
class ArtifactSize:
    path: str
    size_bytes: int


@dataclass(frozen=True)
class PerformanceBenchmarkReport:
    benchmark_name: str
    requests_per_task: int
    top_k: int
    summaries: dict[str, PerformanceSummary]
    artifacts: list[ArtifactSize]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_performance_benchmark(
    *,
    benchmark_name: str = "sample",
    requests_per_task: int = 50,
    top_k: int = 5,
    products: pl.DataFrame | None = None,
    interactions: pl.DataFrame | None = None,
    artifact_paths: list[Path] | None = None,
) -> PerformanceBenchmarkReport:
    if requests_per_task <= 0:
        raise ValueError("requests_per_task must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    benchmark = build_benchmark_dataset(benchmark_name)
    catalog = products if products is not None else benchmark.products
    events = interactions if interactions is not None else benchmark.interactions
    search_queries = [case.query for case in benchmark.search_cases]
    recommendation_user_ids = [case.user_id for case in benchmark.recommendation_cases]

    return _run_performance_benchmark_for_frames(
        benchmark_name=benchmark.name,
        products=catalog,
        interactions=events,
        search_queries=search_queries,
        recommendation_user_ids=recommendation_user_ids,
        requests_per_task=requests_per_task,
        top_k=top_k,
        artifact_paths=artifact_paths,
    )


def run_processed_dataset_performance_benchmark(
    dataset_name: str,
    *,
    requests_per_task: int = 50,
    top_k: int = 5,
    dataset_paths: DatasetPaths | None = None,
    artifact_paths: list[Path] | None = None,
) -> PerformanceBenchmarkReport:
    if requests_per_task <= 0:
        raise ValueError("requests_per_task must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    paths = dataset_paths or get_dataset_paths(dataset_name)
    products, interactions = load_processed_dataset(paths)
    search_queries = _search_queries_from_products(products)
    recommendation_user_ids = _user_ids_from_interactions(interactions)

    return _run_performance_benchmark_for_frames(
        benchmark_name=paths.dataset_name,
        products=products,
        interactions=interactions,
        search_queries=search_queries,
        recommendation_user_ids=recommendation_user_ids,
        requests_per_task=requests_per_task,
        top_k=top_k,
        artifact_paths=artifact_paths,
    )


def _run_performance_benchmark_for_frames(
    *,
    benchmark_name: str,
    products: pl.DataFrame,
    interactions: pl.DataFrame,
    search_queries: list[str],
    recommendation_user_ids: list[str],
    requests_per_task: int,
    top_k: int,
    artifact_paths: list[Path] | None,
) -> PerformanceBenchmarkReport:
    if not search_queries:
        raise ValueError("at least one search query is required")
    if not recommendation_user_ids:
        raise ValueError("at least one recommendation user is required")

    search_service = SearchService(products=products)
    recommendation_service = RecommendationService(products=products, interactions=interactions)

    search_measurements = [
        _measure_search_request(
            search_service,
            query=search_queries[index % len(search_queries)],
            top_k=top_k,
        )
        for index in range(requests_per_task)
    ]
    recommendation_measurements = [
        _measure_recommendation_request(
            recommendation_service,
            user_id=recommendation_user_ids[index % len(recommendation_user_ids)],
            top_k=top_k,
        )
        for index in range(requests_per_task)
    ]

    summaries = {
        "search": summarize_measurements("search", search_measurements),
        "recommendation": summarize_measurements(
            "recommendation",
            recommendation_measurements,
        ),
    }

    return PerformanceBenchmarkReport(
        benchmark_name=benchmark_name,
        requests_per_task=requests_per_task,
        top_k=top_k,
        summaries=summaries,
        artifacts=collect_artifact_sizes(artifact_paths or []),
    )


def summarize_measurements(
    task: str,
    measurements: list[RequestMeasurement],
) -> PerformanceSummary:
    if not measurements:
        raise ValueError("at least one measurement is required")

    latencies = sorted(measurement.latency_ms for measurement in measurements)
    request_count = len(measurements)
    error_count = sum(1 for measurement in measurements if measurement.failed)
    empty_result_count = sum(
        1
        for measurement in measurements
        if not measurement.failed and measurement.result_count == 0
    )
    total_latency_ms = sum(latencies)

    return PerformanceSummary(
        task=task,
        request_count=request_count,
        error_count=error_count,
        empty_result_count=empty_result_count,
        total_latency_ms=total_latency_ms,
        average_latency_ms=total_latency_ms / request_count,
        min_latency_ms=latencies[0],
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        p99_latency_ms=percentile(latencies, 99),
        max_latency_ms=latencies[-1],
        throughput_requests_per_second=_throughput(request_count, total_latency_ms),
        error_rate=error_count / request_count,
        empty_result_rate=empty_result_count / request_count,
    )


def percentile(sorted_values: list[float], percentile_rank: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if percentile_rank < 0 or percentile_rank > 100:
        raise ValueError("percentile_rank must be between 0 and 100")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * (percentile_rank / 100)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return sorted_values[lower_index] + (
        sorted_values[upper_index] - sorted_values[lower_index]
    ) * fraction


def collect_artifact_sizes(paths: list[Path]) -> list[ArtifactSize]:
    artifacts: list[ArtifactSize] = []

    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            artifacts.append(ArtifactSize(path=str(path), size_bytes=path.stat().st_size))
            continue

        size_bytes = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
        artifacts.append(ArtifactSize(path=str(path), size_bytes=size_bytes))

    return artifacts


def _search_queries_from_products(products: pl.DataFrame) -> list[str]:
    if "title" not in products.columns:
        raise ValueError("processed products must include a title column")

    queries = [
        str(title)
        for title in products["title"].drop_nulls().unique(maintain_order=True).to_list()
        if str(title).strip()
    ]
    if not queries:
        raise ValueError("processed products must contain at least one title")

    return queries


def _user_ids_from_interactions(interactions: pl.DataFrame) -> list[str]:
    if "user_id" not in interactions.columns:
        raise ValueError("processed interactions must include a user_id column")

    user_ids = [
        str(user_id)
        for user_id in interactions["user_id"].drop_nulls().unique(maintain_order=True).to_list()
        if str(user_id).strip()
    ]
    if not user_ids:
        raise ValueError("processed interactions must contain at least one user_id")

    return user_ids


def write_performance_json_report(
    report: PerformanceBenchmarkReport,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def write_performance_markdown_report(
    report: PerformanceBenchmarkReport,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "| Task | Requests | Errors | Empty Results | Avg ms | p50 ms | p95 ms | p99 ms | RPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows.extend(_summary_row(summary) for summary in report.summaries.values())

    artifact_rows = [
        "| Artifact | Size Bytes |",
        "| --- | ---: |",
        *[
            f"| `{artifact.path}` | {artifact.size_bytes} |"
            for artifact in report.artifacts
        ],
    ]
    if not report.artifacts:
        artifact_rows.append("| No artifact paths supplied | 0 |")

    path.write_text(
        "\n".join(
            [
                "# CommerceMind Performance Benchmark",
                "",
                f"Benchmark: `{report.benchmark_name}`",
                f"Requests per task: `{report.requests_per_task}`",
                f"Top K: `{report.top_k}`",
                "",
                "## Latency And Throughput",
                "",
                *rows,
                "",
                "## Artifact Sizes",
                "",
                *artifact_rows,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _measure_search_request(
    service: SearchService,
    *,
    query: str,
    top_k: int,
) -> RequestMeasurement:
    started_at = perf_counter()
    try:
        response = service.search(SearchRequest(query=query, top_k=top_k))
    except Exception:
        return RequestMeasurement(
            task="search",
            latency_ms=_elapsed_ms(started_at),
            result_count=0,
            failed=True,
        )

    return RequestMeasurement(
        task="search",
        latency_ms=_elapsed_ms(started_at),
        result_count=len(response.results),
        failed=False,
    )


def _measure_recommendation_request(
    service: RecommendationService,
    *,
    user_id: str,
    top_k: int,
) -> RequestMeasurement:
    started_at = perf_counter()
    try:
        response = service.recommend(RecommendationRequest(user_id=user_id, top_k=top_k))
    except Exception:
        return RequestMeasurement(
            task="recommendation",
            latency_ms=_elapsed_ms(started_at),
            result_count=0,
            failed=True,
        )

    return RequestMeasurement(
        task="recommendation",
        latency_ms=_elapsed_ms(started_at),
        result_count=len(response.results),
        failed=False,
    )


def _summary_row(summary: PerformanceSummary) -> str:
    return (
        f"| {summary.task} | "
        f"{summary.request_count} | "
        f"{summary.error_count} | "
        f"{summary.empty_result_count} | "
        f"{summary.average_latency_ms:.3f} | "
        f"{summary.p50_latency_ms:.3f} | "
        f"{summary.p95_latency_ms:.3f} | "
        f"{summary.p99_latency_ms:.3f} | "
        f"{summary.throughput_requests_per_second:.3f} |"
    )


def _throughput(request_count: int, total_latency_ms: float) -> float:
    if total_latency_ms <= 0:
        return 0.0
    return request_count / (total_latency_ms / 1000)


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000

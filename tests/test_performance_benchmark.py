import json
from pathlib import Path

import pytest

from commercemind.data.datasets import DatasetPaths
from commercemind.data.synthetic import SyntheticDatasetSpec, materialize_synthetic_dataset
from commercemind.evaluation.performance import (
    RequestMeasurement,
    collect_artifact_sizes,
    percentile,
    run_performance_benchmark,
    run_processed_dataset_performance_benchmark,
    summarize_measurements,
    write_performance_json_report,
    write_performance_markdown_report,
)
from commercemind.evaluation.run_performance_benchmark import (
    generate_performance_reports,
    main,
)


def test_percentile_interpolates_sorted_values() -> None:
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 50) == 25.0
    assert percentile(values, 95) == pytest.approx(38.5)


def test_summarize_measurements_calculates_rates_and_latency() -> None:
    measurements = [
        RequestMeasurement(task="search", latency_ms=10.0, result_count=2, failed=False),
        RequestMeasurement(task="search", latency_ms=20.0, result_count=0, failed=False),
        RequestMeasurement(task="search", latency_ms=30.0, result_count=0, failed=True),
    ]

    summary = summarize_measurements("search", measurements)

    assert summary.request_count == 3
    assert summary.error_count == 1
    assert summary.empty_result_count == 1
    assert summary.average_latency_ms == 20.0
    assert summary.p50_latency_ms == 20.0
    assert summary.error_rate == pytest.approx(1 / 3)
    assert summary.empty_result_rate == pytest.approx(1 / 3)
    assert summary.throughput_requests_per_second == pytest.approx(50.0)


def test_run_performance_benchmark_returns_search_and_recommendation_summaries() -> None:
    report = run_performance_benchmark(
        benchmark_name="demo",
        requests_per_task=3,
        top_k=2,
    )

    assert report.benchmark_name == "demo"
    assert report.requests_per_task == 3
    assert set(report.summaries) == {"search", "recommendation"}
    assert report.summaries["search"].request_count == 3
    assert report.summaries["recommendation"].request_count == 3


def test_processed_dataset_performance_benchmark_uses_generated_dataset(
    tmp_path: Path,
) -> None:
    materialize_synthetic_dataset(
        SyntheticDatasetSpec(
            dataset_name="synthetic_perf",
            product_count=20,
            user_count=5,
            interaction_count=40,
            seed=17,
        ),
        processed_dir=tmp_path,
    )
    paths = DatasetPaths(
        dataset_name="synthetic_perf",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path,
    )

    report = run_processed_dataset_performance_benchmark(
        "synthetic_perf",
        dataset_paths=paths,
        requests_per_task=2,
        top_k=2,
    )

    assert report.benchmark_name == "synthetic_perf"
    assert report.summaries["search"].request_count == 2
    assert report.summaries["recommendation"].request_count == 2


def test_performance_reports_write_json_and_markdown(tmp_path: Path) -> None:
    report = run_performance_benchmark(
        benchmark_name="demo",
        requests_per_task=2,
        top_k=2,
    )
    json_path = tmp_path / "performance.json"
    markdown_path = tmp_path / "performance.md"

    write_performance_json_report(report, json_path)
    write_performance_markdown_report(report, markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["benchmark_name"] == "demo"
    assert payload["requests_per_task"] == 2
    assert "CommerceMind Performance Benchmark" in markdown
    assert "| search |" in markdown
    assert "| recommendation |" in markdown


def test_collect_artifact_sizes_handles_files_and_directories(tmp_path: Path) -> None:
    artifact_file = tmp_path / "ranker.json"
    artifact_file.write_text("model", encoding="utf-8")
    artifact_dir = tmp_path / "index"
    artifact_dir.mkdir()
    (artifact_dir / "products.faiss").write_bytes(b"abc")
    (artifact_dir / "documents.json").write_text("{}", encoding="utf-8")

    artifacts = collect_artifact_sizes([artifact_file, artifact_dir, tmp_path / "missing"])

    assert [(artifact.path, artifact.size_bytes) for artifact in artifacts] == [
        (str(artifact_file), 5),
        (str(artifact_dir), 5),
    ]


def test_generate_performance_reports_writes_expected_files(tmp_path: Path) -> None:
    generated = generate_performance_reports(
        output_dir=tmp_path,
        benchmark_name="demo",
        requests_per_task=2,
        top_k=2,
    )

    assert generated.json_path == tmp_path / "performance_benchmark.json"
    assert generated.markdown_path == tmp_path / "performance_benchmark.md"
    assert generated.json_path.exists()
    assert generated.markdown_path.exists()


def test_performance_benchmark_cli_writes_reports(tmp_path: Path) -> None:
    exit_code = main(
        [
            "--benchmark",
            "demo",
            "--requests",
            "2",
            "--top-k",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "performance_benchmark.json").exists()
    assert (tmp_path / "performance_benchmark.md").exists()

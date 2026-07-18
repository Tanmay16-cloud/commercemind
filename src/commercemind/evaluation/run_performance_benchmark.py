from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from commercemind.config import get_settings
from commercemind.evaluation.performance import (
    run_performance_benchmark,
    run_processed_dataset_performance_benchmark,
    write_performance_json_report,
    write_performance_markdown_report,
)
from commercemind.storage.load_dataset import DatasetLoadError


@dataclass(frozen=True)
class GeneratedPerformanceReports:
    json_path: Path
    markdown_path: Path


def generate_performance_reports(
    *,
    output_dir: Path | None = None,
    benchmark_name: str = "sample",
    dataset_name: str | None = None,
    requests_per_task: int = 50,
    top_k: int = 5,
    artifact_paths: list[Path] | None = None,
) -> GeneratedPerformanceReports:
    if requests_per_task <= 0:
        raise ValueError("requests_per_task must be positive")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    report_dir = output_dir or _default_output_dir()
    if dataset_name is not None:
        report = run_processed_dataset_performance_benchmark(
            dataset_name,
            requests_per_task=requests_per_task,
            top_k=top_k,
            artifact_paths=artifact_paths,
        )
    else:
        report = run_performance_benchmark(
            benchmark_name=benchmark_name,
            requests_per_task=requests_per_task,
            top_k=top_k,
            artifact_paths=artifact_paths,
        )
    json_path = report_dir / "performance_benchmark.json"
    markdown_path = report_dir / "performance_benchmark.md"

    write_performance_json_report(report, json_path)
    write_performance_markdown_report(report, markdown_path)

    return GeneratedPerformanceReports(
        json_path=json_path,
        markdown_path=markdown_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CommerceMind in-process performance benchmarks."
    )
    parser.add_argument(
        "--benchmark",
        choices=["demo", "sample"],
        default="sample",
        help="Named benchmark dataset to use.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Processed dataset name under data/processed/. Overrides --benchmark.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=50,
        help="Number of requests to run for each task.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of search/recommendation results requested.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where benchmark reports should be written.",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        action="append",
        default=None,
        help="Optional artifact file or directory to include in size reporting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        generated = generate_performance_reports(
            output_dir=args.output_dir,
            benchmark_name=args.benchmark,
            dataset_name=args.dataset,
            requests_per_task=args.requests,
            top_k=args.top_k,
            artifact_paths=args.artifact_path,
        )
    except (DatasetLoadError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Wrote JSON report: {generated.json_path}")
    print(f"Wrote Markdown report: {generated.markdown_path}")
    return 0


def _default_output_dir() -> Path:
    reports_dir = get_settings().reports_dir
    if reports_dir is None:
        raise ValueError("reports_dir must be configured")
    return reports_dir / "performance"


if __name__ == "__main__":
    raise SystemExit(main())

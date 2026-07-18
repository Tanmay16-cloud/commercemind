from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from commercemind.config import get_settings
from commercemind.evaluation.experiments import (
    run_default_offline_experiment,
    run_processed_recommendation_experiment,
    write_experiment_json_report,
    write_experiment_markdown_report,
)
from commercemind.storage.load_dataset import DatasetLoadError


@dataclass(frozen=True)
class GeneratedExperimentReports:
    json_path: Path
    markdown_path: Path


def generate_default_experiment_reports(
    *,
    output_dir: Path | None = None,
    benchmark_name: str = "sample",
    k: int = 2,
) -> GeneratedExperimentReports:
    if k <= 0:
        raise ValueError("k must be positive")

    report_dir = output_dir or get_settings().reports_dir
    if report_dir is None:
        raise ValueError("reports_dir must be configured")

    report = run_default_offline_experiment(benchmark_name=benchmark_name, k=k)
    json_path = report_dir / "offline_experiment.json"
    markdown_path = report_dir / "offline_experiment.md"

    write_experiment_json_report(report, json_path)
    write_experiment_markdown_report(report, markdown_path)

    return GeneratedExperimentReports(
        json_path=json_path,
        markdown_path=markdown_path,
    )


def generate_processed_recommendation_reports(
    *,
    output_dir: Path | None = None,
    dataset_name: str,
    k: int = 5,
) -> GeneratedExperimentReports:
    if k <= 0:
        raise ValueError("k must be positive")

    report_dir = output_dir or get_settings().reports_dir
    if report_dir is None:
        raise ValueError("reports_dir must be configured")

    report = run_processed_recommendation_experiment(dataset_name, k=k)
    json_path = report_dir / "offline_experiment.json"
    markdown_path = report_dir / "offline_experiment.md"

    write_experiment_json_report(report, json_path)
    write_experiment_markdown_report(report, markdown_path)

    return GeneratedExperimentReports(
        json_path=json_path,
        markdown_path=markdown_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CommerceMind offline search and recommendation experiments."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Cutoff used for Precision@K, Recall@K, HitRate@K, and MRR@K.",
    )
    parser.add_argument(
        "--benchmark",
        choices=["demo", "sample"],
        default="sample",
        help="Named benchmark dataset to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Processed dataset name under data/processed/. Overrides --benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where experiment reports should be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.dataset is not None:
            generated = generate_processed_recommendation_reports(
                output_dir=args.output_dir,
                dataset_name=args.dataset,
                k=args.k,
            )
        else:
            generated = generate_default_experiment_reports(
                output_dir=args.output_dir,
                benchmark_name=args.benchmark,
                k=args.k,
            )
    except (DatasetLoadError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Wrote JSON report: {generated.json_path}")
    print(f"Wrote Markdown report: {generated.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

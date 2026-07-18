import json
from pathlib import Path

import polars as pl

from commercemind.data.datasets import DatasetPaths
from commercemind.data.synthetic import SyntheticDatasetSpec, materialize_synthetic_dataset
from commercemind.evaluation.experiments import (
    build_recommendation_holdout_split,
    run_default_offline_experiment,
    run_processed_recommendation_experiment,
    write_experiment_json_report,
    write_experiment_markdown_report,
)


def test_run_default_offline_experiment_compares_systems() -> None:
    report = run_default_offline_experiment(k=2)

    assert report.benchmark_name == "sample"
    assert set(report.search_reports) == {
        "lexical",
        "vector",
        "hybrid",
        "hybrid_ranked",
        "learned_ranked",
    }
    assert set(report.recommendation_reports) == {"personalized"}
    assert len(report.comparison_rows()) == 6


def test_build_recommendation_holdout_split_hides_latest_new_item() -> None:
    interactions = pl.DataFrame(
        {
            "user_id": ["user-1", "user-1", "user-1", "user-2"],
            "item_id": ["item-a", "item-b", "item-c", "item-z"],
            "event_type": ["view", "click", "purchase", "view"],
            "timestamp_ms": [1, 2, 3, 1],
        }
    )

    split = build_recommendation_holdout_split(interactions)

    assert split.skipped_user_count == 1
    assert split.cases[0].user_id == "user-1"
    assert split.cases[0].relevant_item_ids == {"item-c"}
    assert split.training_interactions["item_id"].to_list() == ["item-a", "item-b"]


def test_run_processed_recommendation_experiment_evaluates_holdout_users(
    tmp_path: Path,
) -> None:
    materialize_synthetic_dataset(
        SyntheticDatasetSpec(
            dataset_name="synthetic_eval",
            product_count=30,
            user_count=5,
            interaction_count=60,
            seed=23,
        ),
        processed_dir=tmp_path,
    )
    paths = DatasetPaths(
        dataset_name="synthetic_eval",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path,
    )

    report = run_processed_recommendation_experiment(
        "synthetic_eval",
        dataset_paths=paths,
        k=5,
    )

    assert report.benchmark_name == "synthetic_eval"
    assert set(report.recommendation_reports) == {"personalized_holdout"}
    assert report.metadata is not None
    assert report.metadata["product_count"] == 30
    assert report.metadata["interaction_count"] == 60
    assert report.metadata["evaluated_user_count"] > 0
    assert len(report.comparison_rows()) == 1


def test_write_experiment_reports_create_json_and_markdown(tmp_path: Path) -> None:
    report = run_default_offline_experiment(k=2)
    json_path = tmp_path / "experiment.json"
    markdown_path = tmp_path / "experiment.md"

    write_experiment_json_report(report, json_path)
    write_experiment_markdown_report(report, markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["benchmark_name"] == "sample"
    assert payload["k"] == 2
    assert payload["comparison"][0]["task"] == "search"
    assert "CommerceMind Offline Evaluation" in markdown
    assert "Benchmark: `sample`" in markdown
    assert "| search | hybrid_ranked |" in markdown
    assert "| search | learned_ranked |" in markdown

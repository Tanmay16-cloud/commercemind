import json
from pathlib import Path

from commercemind.data.datasets import DatasetPaths
from commercemind.data.synthetic import SyntheticDatasetSpec, materialize_synthetic_dataset
from commercemind.evaluation.run_experiment import (
    generate_default_experiment_reports,
    generate_processed_recommendation_reports,
    main,
)


def test_generate_default_experiment_reports_writes_expected_files(tmp_path: Path) -> None:
    generated = generate_default_experiment_reports(
        output_dir=tmp_path,
        benchmark_name="demo",
        k=2,
    )

    assert generated.json_path == tmp_path / "offline_experiment.json"
    assert generated.markdown_path == tmp_path / "offline_experiment.md"
    assert generated.json_path.exists()
    assert generated.markdown_path.exists()


def test_run_experiment_cli_writes_comparison_reports(tmp_path: Path) -> None:
    exit_code = main(["--k", "2", "--benchmark", "sample", "--output-dir", str(tmp_path)])

    payload = json.loads((tmp_path / "offline_experiment.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "offline_experiment.md").read_text(encoding="utf-8")

    assert exit_code == 0
    assert payload["benchmark_name"] == "sample"
    assert payload["k"] == 2
    assert len(payload["comparison"]) == 6
    assert "Benchmark: `sample`" in markdown
    assert "| recommendation | personalized |" in markdown


def test_generate_processed_recommendation_reports_writes_holdout_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_dir = tmp_path / "processed" / "synthetic_cli_eval"
    materialize_synthetic_dataset(
        SyntheticDatasetSpec(
            dataset_name="synthetic_cli_eval",
            product_count=30,
            user_count=5,
            interaction_count=60,
            seed=31,
        ),
        processed_dir=dataset_dir,
    )
    paths = DatasetPaths(
        dataset_name="synthetic_cli_eval",
        raw_dir=tmp_path / "raw" / "synthetic_cli_eval",
        processed_dir=dataset_dir,
    )

    monkeypatch.setattr(
        "commercemind.evaluation.experiments.get_dataset_paths",
        lambda dataset_name: paths,
    )

    generated = generate_processed_recommendation_reports(
        output_dir=tmp_path / "reports",
        dataset_name="synthetic_cli_eval",
        k=5,
    )

    payload = json.loads(generated.json_path.read_text(encoding="utf-8"))
    markdown = generated.markdown_path.read_text(encoding="utf-8")

    assert payload["benchmark_name"] == "synthetic_cli_eval"
    assert payload["metadata"]["evaluation_type"] == "processed_recommendation_holdout"
    assert payload["comparison"][0]["system_name"] == "personalized_holdout"
    assert "| recommendation | personalized_holdout |" in markdown

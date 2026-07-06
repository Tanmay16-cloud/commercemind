import json
from pathlib import Path

from commercemind.evaluation.experiments import (
    run_default_offline_experiment,
    write_experiment_json_report,
    write_experiment_markdown_report,
)


def test_run_default_offline_experiment_compares_systems() -> None:
    report = run_default_offline_experiment(k=2)

    assert set(report.search_reports) == {"lexical", "vector", "hybrid", "hybrid_ranked"}
    assert set(report.recommendation_reports) == {"personalized"}
    assert len(report.comparison_rows()) == 5


def test_write_experiment_reports_create_json_and_markdown(tmp_path: Path) -> None:
    report = run_default_offline_experiment(k=2)
    json_path = tmp_path / "experiment.json"
    markdown_path = tmp_path / "experiment.md"

    write_experiment_json_report(report, json_path)
    write_experiment_markdown_report(report, markdown_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert payload["k"] == 2
    assert payload["comparison"][0]["task"] == "search"
    assert "CommerceMind Offline Evaluation" in markdown
    assert "| search | hybrid_ranked |" in markdown

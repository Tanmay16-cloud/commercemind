from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_github_actions_ci_workflow_has_quality_gates() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    expected_fragments = [
        "name: CI",
        "pull_request:",
        "workflow_dispatch:",
        "actions/checkout@v4",
        "actions/setup-python@v5",
        'python-version: "3.12"',
        'python -m pip install -e ".[dev]"',
        "ruff check src tests",
        "mkdir -p work",
        "python -m pytest --basetemp work/pytest",
        "python -m commercemind.evaluation.run_experiment --benchmark demo --k 2",
        "python -m commercemind.evaluation.run_performance_benchmark --benchmark demo",
        "python -m commercemind.data.generate_synthetic_dataset --dataset synthetic_ci",
        "python -m commercemind.evaluation.run_experiment --dataset synthetic_ci",
        "python -m commercemind.evaluation.run_performance_benchmark --dataset synthetic_ci",
        "python -m commercemind.ranking.training --benchmark demo",
        "python -m commercemind.retrieval.build_index --source sample",
        "docker compose config --quiet",
    ]

    for fragment in expected_fragments:
        assert fragment in workflow


def test_ci_docs_describe_local_equivalent_commands() -> None:
    docs = (PROJECT_ROOT / "docs" / "ci.md").read_text(encoding="utf-8")

    assert ".github/workflows/ci.yml" in docs
    assert "ruff check src tests" in docs
    assert "python -m pytest" in docs
    assert "run_experiment --benchmark demo" in docs
    assert "run_performance_benchmark --benchmark demo" in docs
    assert "generate_synthetic_dataset --dataset synthetic_ci" in docs
    assert "run_experiment --dataset synthetic_ci" in docs
    assert "run_performance_benchmark --dataset synthetic_ci" in docs
    assert "ranking.training --benchmark demo" in docs
    assert "retrieval.build_index --source sample" in docs
    assert "docker compose config --quiet" in docs

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_readme_links_final_showcase_documentation() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    expected_links = [
        "docs/project_showcase.md",
        "docs/results.md",
        "docs/datasets.md",
        "docs/synthetic_dataset.md",
        "docs/performance.md",
        "docs/monitoring.md",
        "docs/deployment.md",
        "docs/ci.md",
    ]

    for link in expected_links:
        assert link in readme


def test_project_showcase_contains_resume_and_architecture_sections() -> None:
    showcase = (PROJECT_ROOT / "docs" / "project_showcase.md").read_text(
        encoding="utf-8"
    )

    assert "One-Line Summary" in showcase
    assert "System Architecture" in showcase
    assert "Resume Bullets" in showcase
    assert "Interview Talking Points" in showcase
    assert "Demo Flow" in showcase


def test_results_snapshot_contains_quality_and_system_metrics() -> None:
    results = (PROJECT_ROOT / "docs" / "results.md").read_text(encoding="utf-8")

    assert "Precision@K" in results
    assert "Recall@K" in results
    assert "HitRate@K" in results
    assert "MRR@K" in results
    assert "p95 ms" in results
    assert "synthetic_large" in results
    assert "HitRate@5" in results
    assert "97 passed" in results
    assert "91% statement coverage" in results

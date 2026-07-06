import json
from pathlib import Path

import pytest

from commercemind.evaluation.recommendation import (
    RecommendationEvaluationCase,
    evaluate_recommender,
    write_recommendation_report,
)
from commercemind.services.recommendations import RecommendationService


def test_evaluate_recommender_summarizes_case_metrics() -> None:
    recommender = RecommendationService()
    cases = [
        RecommendationEvaluationCase(
            user_id="user-runner",
            relevant_item_ids={"sku-trail-shoes-001"},
        )
    ]

    report = evaluate_recommender(recommender, cases, k=2)

    assert report.summary.case_count == 1
    assert report.summary.k == 2
    assert report.summary.mean_hit_rate_at_k == 1.0
    assert report.summary.mean_recall_at_k == 1.0
    assert report.summary.mean_reciprocal_rank_at_k == 1.0


def test_write_recommendation_report_writes_json(tmp_path: Path) -> None:
    report = evaluate_recommender(
        RecommendationService(),
        [
            RecommendationEvaluationCase(
                user_id="user-runner",
                relevant_item_ids={"sku-trail-shoes-001"},
            )
        ],
        k=1,
    )
    output_path = tmp_path / "recommendations.json"

    write_recommendation_report(report, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["case_count"] == 1
    assert payload["cases"][0]["recommended_item_ids"] == ["sku-trail-shoes-001"]


def test_evaluate_recommender_rejects_empty_cases() -> None:
    with pytest.raises(ValueError, match="at least one evaluation case"):
        evaluate_recommender(RecommendationService(), [], k=1)

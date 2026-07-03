import json
from pathlib import Path

import polars as pl
import pytest

from commercemind.evaluation.retrieval import (
    RetrievalEvaluationCase,
    evaluate_search_retriever,
    write_retrieval_report,
)
from commercemind.retrieval.baseline import LexicalRetriever


def test_evaluate_search_retriever_summarizes_case_metrics() -> None:
    retriever = LexicalRetriever(
        pl.DataFrame(
            {
                "item_id": ["A1", "A2", "A3"],
                "title": ["Running Shoes", "Trail Shoes", "Office Shirt"],
                "category": ["Footwear", "Footwear", "Apparel"],
                "brand": ["Nike", "Acme", "Acme"],
                "description": [
                    "Road running shoes",
                    "Shoes for trail running",
                    "Formal shirt",
                ],
                "price": [2999.0, 3499.0, 1499.0],
            }
        )
    )
    cases = [
        RetrievalEvaluationCase(query="running shoes", relevant_item_ids={"A1", "A2"}),
        RetrievalEvaluationCase(query="formal shirt", relevant_item_ids={"A3"}),
    ]

    report = evaluate_search_retriever(retriever, cases, k=2)

    assert report.summary.case_count == 2
    assert report.summary.k == 2
    assert report.summary.mean_hit_rate_at_k == 1.0
    assert report.summary.mean_recall_at_k == pytest.approx(1.0)
    assert report.summary.mean_reciprocal_rank_at_k == 1.0


def test_write_retrieval_report_writes_json(tmp_path: Path) -> None:
    retriever = LexicalRetriever(
        pl.DataFrame(
            {
                "item_id": ["A1"],
                "title": ["Running Shoes"],
                "category": ["Footwear"],
                "brand": ["Nike"],
                "description": ["Road running shoes"],
                "price": [2999.0],
            }
        )
    )
    report = evaluate_search_retriever(
        retriever,
        [RetrievalEvaluationCase(query="running", relevant_item_ids={"A1"})],
        k=1,
    )
    output_path = tmp_path / "reports" / "retrieval.json"

    write_retrieval_report(report, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["case_count"] == 1
    assert payload["cases"][0]["retrieved_item_ids"] == ["A1"]


def test_evaluate_search_retriever_rejects_empty_cases() -> None:
    retriever = LexicalRetriever(
        pl.DataFrame(
            {
                "item_id": ["A1"],
                "title": ["Running Shoes"],
                "category": ["Footwear"],
                "brand": ["Nike"],
                "description": ["Road running shoes"],
                "price": [2999.0],
            }
        )
    )

    with pytest.raises(ValueError, match="at least one evaluation case"):
        evaluate_search_retriever(retriever, [], k=1)

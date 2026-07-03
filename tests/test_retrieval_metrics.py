import pytest

from commercemind.retrieval.metrics import (
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_precision_at_k_counts_hits_in_retrieved_prefix() -> None:
    assert precision_at_k({"A1", "A3"}, ["A1", "A2", "A3"], k=2) == 0.5


def test_recall_at_k_counts_fraction_of_relevant_items_found() -> None:
    assert recall_at_k({"A1", "A3"}, ["A1", "A2", "A3"], k=2) == 0.5


def test_hit_rate_at_k_returns_one_when_any_relevant_item_is_found() -> None:
    assert hit_rate_at_k({"A3"}, ["A1", "A2", "A3"], k=3) == 1.0


def test_reciprocal_rank_at_k_returns_inverse_first_relevant_rank() -> None:
    assert reciprocal_rank_at_k({"A3"}, ["A1", "A2", "A3"], k=3) == pytest.approx(1 / 3)


def test_metrics_reject_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        precision_at_k({"A1"}, ["A1"], k=0)

from pathlib import Path

import polars as pl

from commercemind.evaluation.retrieval import RetrievalEvaluationCase
from commercemind.ranking.learned import (
    LearnedLinearProductRanker,
    load_learned_ranking_model,
    save_learned_ranking_model,
)
from commercemind.ranking.training import (
    build_ranking_training_examples,
    train_learned_ranking_model,
)
from commercemind.retrieval.baseline import RetrievalCandidate


def test_train_learned_ranker_promotes_relevant_candidate() -> None:
    products = _products()
    examples = build_ranking_training_examples(
        products,
        [
            RetrievalEvaluationCase(
                query="running shoes",
                relevant_item_ids={"A1"},
            )
        ],
    )
    model = train_learned_ranking_model(examples, l2_regularization=0.1)
    ranker = LearnedLinearProductRanker(products, model)

    ranked = ranker.rank(
        "running shoes",
        [
            RetrievalCandidate(item_id="A2", title="Office Shirt", score=0.1),
            RetrievalCandidate(item_id="A1", title="Running Shoes", score=0.1),
        ],
        top_k=2,
    )

    assert ranked[0].item_id == "A1"
    assert ranked[0].score > ranked[1].score


def test_learned_ranking_model_round_trips_to_disk(tmp_path: Path) -> None:
    products = _products()
    examples = build_ranking_training_examples(
        products,
        [
            RetrievalEvaluationCase(
                query="running shoes",
                relevant_item_ids={"A1"},
            )
        ],
    )
    model = train_learned_ranking_model(examples, l2_regularization=0.1)
    path = tmp_path / "ranker.json"

    save_learned_ranking_model(model, path)
    loaded = load_learned_ranking_model(path)

    assert path.exists()
    assert loaded.feature_names == model.feature_names
    assert loaded.weights == model.weights
    assert loaded.bias == model.bias


def _products() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "item_id": ["A1", "A2"],
            "title": ["Running Shoes", "Office Shirt"],
            "category": ["Footwear", "Apparel"],
            "brand": ["StrideLab", "Acme"],
            "description": ["Daily training shoes", "Formal shirt for work"],
            "price": [2999.0, 1499.0],
        }
    )

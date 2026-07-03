import polars as pl

from commercemind.retrieval.baseline import LexicalRetriever, PopularityRetriever, tokenize


def test_tokenize_lowercases_and_extracts_words() -> None:
    assert tokenize("Running Shoes 2.0!") == ["running", "shoes", "2", "0"]


def test_lexical_retriever_scores_matching_products() -> None:
    products = pl.DataFrame(
        {
            "item_id": ["A1", "A2"],
            "title": ["Running Shoes", "Office Shirt"],
            "category": ["Footwear", "Apparel"],
            "brand": ["Nike", "Acme"],
            "description": ["Shoes for daily running", "Formal shirt for work"],
            "price": [2999.0, 1499.0],
        }
    )

    retriever = LexicalRetriever(products)
    candidates = retriever.retrieve("running shoes", top_k=2)

    assert [candidate.item_id for candidate in candidates] == ["A1"]
    assert candidates[0].score > 0


def test_popularity_retriever_orders_by_interaction_count() -> None:
    products = pl.DataFrame(
        {
            "item_id": ["A1", "A2"],
            "title": ["Running Shoes", "Office Shirt"],
        }
    )
    interactions = pl.DataFrame(
        {
            "user_id": ["U1", "U2", "U3"],
            "item_id": ["A2", "A2", "A1"],
            "event_type": ["click", "purchase", "click"],
            "timestamp_ms": [1, 2, 3],
        }
    )

    retriever = PopularityRetriever(products, interactions)
    candidates = retriever.retrieve(top_k=2)

    assert [candidate.item_id for candidate in candidates] == ["A2", "A1"]
    assert candidates[0].score == 2.0

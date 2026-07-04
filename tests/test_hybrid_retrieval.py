from commercemind.retrieval.baseline import RetrievalCandidate
from commercemind.retrieval.hybrid import HybridRetriever, RetrieverWeight


class StubRetriever:
    def __init__(self, candidates: list[RetrievalCandidate]) -> None:
        self._candidates = candidates

    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        return self._candidates[:top_k]


def test_hybrid_retriever_boosts_items_seen_by_multiple_retrievers() -> None:
    lexical = StubRetriever(
        [
            RetrievalCandidate(item_id="A1", title="Running Shoes", score=10.0),
            RetrievalCandidate(item_id="A2", title="Gym Shirt", score=5.0),
        ]
    )
    vector = StubRetriever(
        [
            RetrievalCandidate(item_id="A3", title="Trail Sneakers", score=0.9),
            RetrievalCandidate(item_id="A1", title="Running Shoes", score=0.8),
        ]
    )

    retriever = HybridRetriever([lexical, vector], rrf_k=1, candidate_pool_size=10)
    candidates = retriever.retrieve("running shoes", top_k=3)

    assert [candidate.item_id for candidate in candidates] == ["A1", "A3", "A2"]
    assert candidates[0].score > candidates[1].score


def test_hybrid_retriever_supports_weighted_sources() -> None:
    lexical = StubRetriever(
        [RetrievalCandidate(item_id="A1", title="Running Shoes", score=10.0)]
    )
    vector = StubRetriever(
        [RetrievalCandidate(item_id="A2", title="Trail Sneakers", score=0.9)]
    )

    retriever = HybridRetriever(
        [
            RetrieverWeight(lexical, weight=1.0),
            RetrieverWeight(vector, weight=3.0),
        ],
        rrf_k=1,
    )
    candidates = retriever.retrieve("sneakers", top_k=2)

    assert [candidate.item_id for candidate in candidates] == ["A2", "A1"]


def test_hybrid_retriever_returns_empty_list_for_non_positive_top_k() -> None:
    retriever = HybridRetriever(
        [StubRetriever([RetrievalCandidate(item_id="A1", title="Running Shoes", score=1.0)])]
    )

    assert retriever.retrieve("running", top_k=0) == []

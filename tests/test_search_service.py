from commercemind.retrieval.baseline import RetrievalCandidate
from commercemind.schemas import SearchRequest
from commercemind.services.search import SearchService


def test_search_service_returns_trimmed_query_and_result() -> None:
    service = SearchService()

    response = service.search(SearchRequest(query="  running shoes  ", top_k=5))

    assert response.query == "running shoes"
    assert len(response.results) >= 1
    assert response.results[0].item_id == "sku-running-shoes-001"
    assert response.results[0].title == "Running Shoes"
    assert response.results[0].score > 0


class SpyRetriever:
    def __init__(self) -> None:
        self.query = ""
        self.top_k = 0

    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        self.query = query
        self.top_k = top_k
        return [
            RetrievalCandidate(item_id="A1", title="Running Shoes", score=0.8),
            RetrievalCandidate(item_id="A2", title="Gym Shirt", score=0.7),
        ]


class ReverseRanker:
    def rank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        return list(reversed(candidates))[:top_k]


def test_search_service_retrieves_candidate_pool_before_ranking() -> None:
    retriever = SpyRetriever()
    service = SearchService(
        retriever=retriever,
        ranker=ReverseRanker(),
        candidate_pool_multiplier=4,
    )

    response = service.search(SearchRequest(query="  running shoes  ", top_k=1))

    assert retriever.query == "running shoes"
    assert retriever.top_k == 4
    assert [result.item_id for result in response.results] == ["A2"]

from commercemind.schemas import SearchRequest
from commercemind.services.search import SearchService


def test_search_service_returns_trimmed_query_and_result() -> None:
    service = SearchService()

    response = service.search(SearchRequest(query="  running shoes  ", top_k=5))

    assert response.query == "running shoes"
    assert len(response.results) == 1
    assert response.results[0].item_id == "sku-running-shoes-001"
    assert response.results[0].title == "Running Shoes"
    assert response.results[0].score > 0

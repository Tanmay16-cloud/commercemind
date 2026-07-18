from fastapi.testclient import TestClient

from commercemind.main import app

client = TestClient(app)


def test_search_endpoint_returns_baseline_result() -> None:
    response = client.post(
        "/search",
        json={
            "query": "  running shoes  ",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["query"] == "running shoes"
    assert len(payload["results"]) >= 1
    assert payload["results"][0]["item_id"] == "sku-running-shoes-001"
    assert payload["results"][0]["title"] == "Running Shoes"
    assert payload["results"][0]["score"] > 0


def test_search_endpoint_rejects_blank_query() -> None:
    response = client.post("/search", json={"query": "   ", "top_k": 3})

    assert response.status_code == 422

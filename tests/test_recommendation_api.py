from fastapi.testclient import TestClient

from commercemind.main import app

client = TestClient(app)


def test_recommendations_endpoint_returns_ranked_results() -> None:
    response = client.post(
        "/recommendations",
        json={
            "user_id": "user-runner",
            "top_k": 2,
        },
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["user_id"] == "user-runner"
    assert len(payload["results"]) == 2
    assert payload["results"][0]["item_id"] == "sku-trail-shoes-001"
    assert payload["results"][0]["score"] > 0

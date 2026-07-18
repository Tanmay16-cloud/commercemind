from fastapi.testclient import TestClient

from commercemind.main import app

client = TestClient(app)


def test_metrics_endpoint_returns_runtime_metrics() -> None:
    before = client.get("/metrics").json()
    previous_search_count = _request_count(before, "search")

    response = client.post(
        "/search",
        json={
            "query": "running shoes",
            "top_k": 2,
        },
    )
    metrics_response = client.get("/metrics")

    assert response.status_code == 200
    assert metrics_response.status_code == 200

    search_metrics = metrics_response.json()["endpoints"]["search"]
    assert search_metrics["request_count"] >= previous_search_count + 1
    assert search_metrics["error_count"] >= 0
    assert search_metrics["empty_result_count"] >= 0
    assert search_metrics["total_latency_ms"] >= 0
    assert search_metrics["average_latency_ms"] >= 0
    assert search_metrics["max_latency_ms"] >= 0
    assert 0 <= search_metrics["error_rate"] <= 1
    assert 0 <= search_metrics["empty_result_rate"] <= 1


def test_metrics_endpoint_records_recommendation_requests() -> None:
    before = client.get("/metrics").json()
    previous_recommendation_count = _request_count(before, "recommendations")

    response = client.post(
        "/recommendations",
        json={
            "user_id": "user-runner",
            "top_k": 2,
        },
    )
    metrics_response = client.get("/metrics")

    assert response.status_code == 200

    recommendation_metrics = metrics_response.json()["endpoints"]["recommendations"]
    assert recommendation_metrics["request_count"] >= previous_recommendation_count + 1
    assert recommendation_metrics["average_latency_ms"] >= 0


def _request_count(payload: dict[str, object], endpoint: str) -> int:
    endpoints = payload.get("endpoints", {})
    if not isinstance(endpoints, dict):
        return 0

    metrics = endpoints.get(endpoint, {})
    if not isinstance(metrics, dict):
        return 0

    return int(metrics.get("request_count", 0))

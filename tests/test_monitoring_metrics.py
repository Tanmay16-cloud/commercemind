import pytest

from commercemind.monitoring.metrics import MetricsRegistry


def test_metrics_registry_records_latency_counts_and_rates() -> None:
    registry = MetricsRegistry()

    registry.record("search", latency_ms=10.0, result_count=2)
    registry.record("search", latency_ms=30.0, result_count=0)
    registry.record("search", latency_ms=20.0, failed=True)

    snapshot = registry.snapshot()["search"]

    assert snapshot.request_count == 3
    assert snapshot.error_count == 1
    assert snapshot.empty_result_count == 1
    assert snapshot.total_latency_ms == 60.0
    assert snapshot.average_latency_ms == 20.0
    assert snapshot.max_latency_ms == 30.0
    assert snapshot.error_rate == pytest.approx(1 / 3)
    assert snapshot.empty_result_rate == pytest.approx(1 / 3)


def test_metrics_registry_rejects_invalid_values() -> None:
    registry = MetricsRegistry()

    with pytest.raises(ValueError, match="latency_ms"):
        registry.record("search", latency_ms=-1.0)

    with pytest.raises(ValueError, match="result_count"):
        registry.record("search", latency_ms=1.0, result_count=-1)

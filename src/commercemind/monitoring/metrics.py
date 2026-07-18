from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class EndpointMetricSnapshot:
    request_count: int
    error_count: int
    empty_result_count: int
    total_latency_ms: float
    average_latency_ms: float
    max_latency_ms: float
    error_rate: float
    empty_result_rate: float


@dataclass
class _EndpointMetricState:
    request_count: int = 0
    error_count: int = 0
    empty_result_count: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    def record(self, *, latency_ms: float, result_count: int, failed: bool) -> None:
        self.request_count += 1
        self.total_latency_ms += latency_ms
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)

        if failed:
            self.error_count += 1
        if not failed and result_count == 0:
            self.empty_result_count += 1

    def snapshot(self) -> EndpointMetricSnapshot:
        average_latency_ms = (
            self.total_latency_ms / self.request_count if self.request_count else 0.0
        )
        error_rate = self.error_count / self.request_count if self.request_count else 0.0
        empty_result_rate = (
            self.empty_result_count / self.request_count if self.request_count else 0.0
        )

        return EndpointMetricSnapshot(
            request_count=self.request_count,
            error_count=self.error_count,
            empty_result_count=self.empty_result_count,
            total_latency_ms=self.total_latency_ms,
            average_latency_ms=average_latency_ms,
            max_latency_ms=self.max_latency_ms,
            error_rate=error_rate,
            empty_result_rate=empty_result_rate,
        )


class MetricsRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, _EndpointMetricState] = {}
        self._lock = Lock()

    def record(
        self,
        endpoint: str,
        *,
        latency_ms: float,
        result_count: int = 0,
        failed: bool = False,
    ) -> None:
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if result_count < 0:
            raise ValueError("result_count must be non-negative")

        with self._lock:
            state = self._metrics.setdefault(endpoint, _EndpointMetricState())
            state.record(
                latency_ms=latency_ms,
                result_count=result_count,
                failed=failed,
            )

    def snapshot(self) -> dict[str, EndpointMetricSnapshot]:
        with self._lock:
            return {
                endpoint: state.snapshot()
                for endpoint, state in sorted(self._metrics.items())
            }

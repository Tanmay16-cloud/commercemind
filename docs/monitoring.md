# Monitoring

CommerceMind exposes lightweight in-process runtime metrics through:

```text
GET /metrics
```

The endpoint returns one metrics object per instrumented API path. Search requests
are grouped under `search`, and recommendation requests are grouped under
`recommendations`.

## Metrics

Each endpoint includes:

- `request_count`: total requests handled by the endpoint.
- `error_count`: requests that raised an exception inside the endpoint.
- `empty_result_count`: successful requests that returned no search or recommendation results.
- `total_latency_ms`: total latency observed across all recorded requests.
- `average_latency_ms`: mean request latency.
- `max_latency_ms`: slowest observed request.
- `error_rate`: `error_count / request_count`.
- `empty_result_rate`: `empty_result_count / request_count`.

## Example

Run the API:

```powershell
python -m uvicorn commercemind.main:app --reload
```

Send a search request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search -ContentType "application/json" -Body '{"query":"running shoes","top_k":5}'
```

Read metrics:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

Example shape:

```json
{
  "endpoints": {
    "search": {
      "request_count": 1,
      "error_count": 0,
      "empty_result_count": 0,
      "total_latency_ms": 12.5,
      "average_latency_ms": 12.5,
      "max_latency_ms": 12.5,
      "error_rate": 0.0,
      "empty_result_rate": 0.0
    }
  }
}
```

## Why This Matters

Offline evaluation tells us whether ranking quality is good before deployment.
Runtime monitoring tells us whether the live API is healthy after deployment.
Together, they help detect regressions such as slow responses, empty search
results, or endpoint failures.

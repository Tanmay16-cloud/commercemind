# Performance Benchmarking

CommerceMind includes an in-process benchmark runner for measuring service-level
latency and throughput without needing to start an HTTP server.

## Run A Benchmark

```powershell
python -m commercemind.evaluation.run_performance_benchmark --benchmark sample --requests 100 --top-k 5 --output-dir reports/performance
```

This writes:

```text
reports/performance/performance_benchmark.json
reports/performance/performance_benchmark.md
```

## Run Against A Processed Dataset

After generating or ingesting a processed dataset, pass `--dataset` instead of
`--benchmark`:

```powershell
python -m commercemind.evaluation.run_performance_benchmark --dataset synthetic_large --requests 25 --top-k 5 --output-dir reports/performance-synthetic
```

The runner loads:

```text
data/processed/synthetic_large/products.parquet
data/processed/synthetic_large/interactions.parquet
```

This is the recommended path when you want to demonstrate behavior beyond the
small built-in benchmark fixtures.

## What It Measures

The benchmark runs repeated search and recommendation requests against the
project services and reports:

- `request_count`: number of measured requests.
- `error_count`: requests that raised an exception.
- `empty_result_count`: successful requests with no returned items.
- `average_latency_ms`: average service latency.
- `p50_latency_ms`: median latency.
- `p95_latency_ms`: latency below which 95% of requests completed.
- `p99_latency_ms`: latency below which 99% of requests completed.
- `max_latency_ms`: slowest request.
- `throughput_requests_per_second`: completed requests per second.
- `error_rate`: failed requests divided by total requests.
- `empty_result_rate`: empty successful responses divided by total requests.

## Artifact Size Reporting

You can include generated artifacts such as a FAISS index or learned ranker:

```powershell
python -m commercemind.evaluation.run_performance_benchmark --benchmark sample --requests 100 --top-k 5 --output-dir reports/performance --artifact-path artifacts/indexes/products --artifact-path artifacts/models/ranker.json
```

The report will include the total byte size of each supplied artifact path.

## Why This Matters

Offline evaluation tells us whether retrieval and ranking quality are good.
Performance benchmarking tells us whether the system is fast enough and stable
enough to serve user traffic.

For interviews, this is useful because it lets you discuss both model quality and
system quality: recall, MRR, p95 latency, throughput, error rate, and artifact
size.

# Results Snapshot

This page records a reproducible snapshot from the local benchmark datasets and
the generated synthetic scale dataset.
Generated reports are written under `reports/`, which is ignored by git, so this
document gives reviewers a stable summary to read on GitHub.

## Offline Evaluation

Command:

```powershell
python -m commercemind.evaluation.run_experiment --benchmark sample --k 2
```

Snapshot:

| Task | System | Precision@K | Recall@K | HitRate@K | MRR@K |
| --- | --- | ---: | ---: | ---: | ---: |
| search | lexical | 0.750 | 1.000 | 1.000 | 1.000 |
| search | vector | 0.750 | 1.000 | 1.000 | 1.000 |
| search | hybrid | 0.750 | 1.000 | 1.000 | 1.000 |
| search | hybrid_ranked | 0.750 | 1.000 | 1.000 | 1.000 |
| search | learned_ranked | 0.750 | 1.000 | 1.000 | 1.000 |
| recommendation | personalized | 0.500 | 1.000 | 1.000 | 0.875 |

## Performance Benchmark

Command:

```powershell
python -m commercemind.evaluation.run_performance_benchmark --benchmark demo --requests 5 --top-k 2 --output-dir reports/performance
```

Example local run:

| Task | Requests | Errors | Empty Results | Avg ms | p50 ms | p95 ms | p99 ms | RPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| search | 5 | 0 | 0 | 0.197 | 0.122 | 0.417 | 0.469 | 5068.424 |
| recommendation | 5 | 0 | 0 | 0.121 | 0.090 | 0.241 | 0.266 | 8233.163 |

Performance values depend on local hardware and environment. Use them as a
sanity-check snapshot, not as universal production latency claims.

## Synthetic Scale Benchmark

Dataset generation command:

```powershell
python -m commercemind.data.generate_synthetic_dataset --dataset synthetic_large --products 1000 --users 100 --interactions 5000 --seed 42
```

Benchmark command:

```powershell
python -m commercemind.evaluation.run_performance_benchmark --dataset synthetic_large --requests 25 --top-k 5 --output-dir reports/performance-synthetic
```

Recommendation holdout command:

```powershell
python -m commercemind.evaluation.run_experiment --dataset synthetic_large --k 5 --output-dir reports/recommendation-synthetic
```

Dataset size:

| Dataset | Products | Users | Interactions |
| --- | ---: | ---: | ---: |
| synthetic_large | 1000 | 100 | 5000 |

Example local run:

| Task | Requests | Errors | Empty Results | Avg ms | p50 ms | p95 ms | p99 ms | RPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| search | 25 | 0 | 0 | 3.669 | 3.426 | 4.868 | 5.150 | 272.584 |
| recommendation | 25 | 0 | 0 | 3.325 | 3.212 | 4.786 | 5.118 | 300.709 |

Processed recommendation holdout:

| Dataset | Evaluated Users | Training Interactions | Precision@5 | Recall@5 | HitRate@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_large | 100 | 3961 | 0.028 | 0.016 | 0.140 | 0.051 |

This holdout evaluation trains the recommender on each user's earlier
interactions and evaluates whether top-5 recommendations recover items from the
user's future interaction window.

## Test Coverage Snapshot

Latest local verification:

```text
97 passed
91% statement coverage
ruff check src tests: passed
mypy src: passed
```

The test suite covers retrieval, vector indexing, ranking, learned ranker
training, recommendations, storage, evaluation, monitoring, performance
benchmarking, synthetic dataset generation, processed-dataset recommendation
holdout evaluation, deployment config, and CI config.

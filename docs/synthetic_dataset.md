# Synthetic Dataset

CommerceMind includes a deterministic synthetic dataset generator so the system
can be benchmarked beyond the tiny demo catalog without committing large data
files to git.

## Why It Exists

The built-in `demo` and `sample` datasets are useful for fast tests and clear
examples, but they are intentionally small. The synthetic generator creates a
larger processed catalog with realistic product fields and user interaction
patterns. This lets you show that retrieval, ranking, recommendations, and
benchmarking work on a more meaningful workload.

## Generate A Larger Dataset

```powershell
python -m commercemind.data.generate_synthetic_dataset --dataset synthetic_large --products 1000 --users 100 --interactions 5000 --seed 42
```

This writes:

```text
data/processed/synthetic_large/products.parquet
data/processed/synthetic_large/interactions.parquet
```

The generated files are ignored by git through `.gitignore`, so reviewers get
the code and commands while local machines can regenerate the data whenever
needed.

## Generated Product Schema

Each product has the same normalized columns used by the rest of the project:

```text
item_id | title | category | brand | description | price
```

Example product shape:

```text
synthetic-item-000001 | Durable CoreStep Running Shoes | Footwear | CoreStep | durable running shoes for footwear shoppers who prefer CoreStep products. | 3199.52
```

## Generated Interaction Schema

Each interaction has:

```text
user_id | item_id | event_type | timestamp_ms
```

Users are given category preferences, so their events are not purely random.
Most interactions come from the user's preferred category, while some events
come from the wider catalog. This gives the recommendation system a useful
personalization signal.

Supported event types:

- `view`
- `click`
- `add_to_cart`
- `purchase`

## Benchmark The Generated Dataset

```powershell
python -m commercemind.evaluation.run_performance_benchmark --dataset synthetic_large --requests 25 --top-k 5 --output-dir reports/performance-synthetic
```

This loads the processed Parquet files, builds the search and recommendation
services in process, runs repeated requests, and writes:

```text
reports/performance-synthetic/performance_benchmark.json
reports/performance-synthetic/performance_benchmark.md
```

## Evaluate Recommendation Accuracy

```powershell
python -m commercemind.evaluation.run_experiment --dataset synthetic_large --k 5 --output-dir reports/recommendation-synthetic
```

This uses a chronological holdout:

```text
earlier user interactions -> recommendation history
future user interactions -> ground truth relevant items
```

The recommender is evaluated with `Precision@5`, `Recall@5`, `HitRate@5`, and
`MRR@5`.

## Current Local Snapshot

The current checked documentation was generated from:

```text
1000 products
100 users
5000 interactions
25 search requests
25 recommendation requests
top_k = 5
```

Result summary:

| Task | Requests | Errors | Empty Results | Avg ms | p95 ms | RPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| search | 25 | 0 | 0 | 3.669 | 4.868 | 272.584 |
| recommendation | 25 | 0 | 0 | 3.325 | 4.786 | 300.709 |

Recommendation holdout summary:

| Evaluated Users | Training Interactions | Precision@5 | Recall@5 | HitRate@5 | MRR@5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 3961 | 0.028 | 0.016 | 0.140 | 0.051 |

## Interview Explanation

The synthetic dataset is not meant to replace a real production dataset. Its
purpose is to prove that the project has a repeatable scale-testing and
holdout-evaluation path. In an interview, you can say:

```text
I kept small deterministic fixtures for unit tests, but also added a synthetic
data generator so the same retrieval, ranking, and recommendation services can
be benchmarked and evaluated on a larger catalog without shipping large data
files.
```

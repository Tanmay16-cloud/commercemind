# CommerceMind

[![CI](https://github.com/Tanmay16-cloud/commercemind/actions/workflows/ci.yml/badge.svg)](https://github.com/Tanmay16-cloud/commercemind/actions/workflows/ci.yml)

CommerceMind is a production-style product discovery backend for e-commerce catalogs. It combines data ingestion, lexical retrieval, configurable Sentence Transformer vector search, hybrid candidate generation, learned ranking, personalized recommendations, offline evaluation, persistence, monitoring, benchmarking, Docker, and CI.

## Highlights

- FastAPI service for `/search`, `/recommendations`, `/health`, and `/metrics`
- Lexical, Sentence Transformer, and hybrid retrieval with Reciprocal Rank Fusion
- Feature-based ranking and lightweight learned ranker training
- Personalized recommendations from user interaction history
- Versioned FAISS indexes with embedding metadata and catalog-fingerprint validation
- SQLAlchemy repositories for SQLite/Postgres-backed products and interactions
- Offline quality reports for Precision@K, Recall@K, HitRate@K, and MRR@K
- Processed-dataset recommendation holdout evaluation from future interactions
- Deterministic synthetic dataset generation for larger local benchmarks
- Runtime monitoring and in-process performance benchmarking
- Docker, Docker Compose, GitHub Actions CI, MyPy, and 97 tests

## Tech Stack

Python 3.12, FastAPI, Pydantic, Polars, NumPy, Sentence Transformers, FAISS,
SQLAlchemy, PostgreSQL/SQLite, Pytest, Ruff, MyPy, Docker, and GitHub Actions.

## Results Snapshot

Current sample benchmark:

| Task | System | Precision@K | Recall@K | HitRate@K | MRR@K |
| --- | --- | ---: | ---: | ---: | ---: |
| search | hybrid_ranked | 0.750 | 1.000 | 1.000 | 1.000 |
| search | learned_ranked | 0.750 | 1.000 | 1.000 | 1.000 |
| recommendation | personalized | 0.500 | 1.000 | 1.000 | 0.875 |

Latest local verification:

```text
97 passed | 91% coverage
ruff check src tests: passed
mypy src: passed
```

## Resume-Ready Highlights

- Built an end-to-end e-commerce search and recommendation backend with hybrid lexical/vector retrieval, Reciprocal Rank Fusion, feature engineering, and learned ranking.
- Designed reproducible offline evaluation and chronological recommendation holdouts; validated the system with 97 automated tests, 91% coverage, static typing, and CI quality gates.
- Benchmarked a deterministic workload of 1,000 products and 5,000 interactions, packaged the API with Docker, and persisted validated FAISS/model artifacts for deployment.

Synthetic scale benchmark:

| Dataset | Products | Users | Interactions | Search p95 ms | Rec p95 ms | Rec HitRate@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic_large | 1000 | 100 | 5000 | 4.868 | 4.786 | 0.140 |

More details are available in [docs/results.md](docs/results.md), [docs/synthetic_dataset.md](docs/synthetic_dataset.md), and [docs/project_showcase.md](docs/project_showcase.md).

## Goals

- Build semantic product search for natural language queries.
- Build personalized recommendations for returning users.
- Build similar-item recommendations for product detail pages.
- Compare simple baselines against learned retrieval and ranking models.
- Serve results through a typed API with measurable latency.
- Track experiments, model versions, evaluation metrics, and drift.

## User Flows

1. A shopper searches for a product using natural language.
2. A returning user receives personalized product recommendations.
3. A product page shows similar and complementary products.
4. An operator can inspect model quality, latency, and example results.

## System Architecture

```text
Raw dataset
-> ingestion and cleaning
-> feature engineering
-> baseline retrieval
-> dense vector retrieval
-> hybrid candidate generation
-> ranking
-> personalized recommendations
-> offline evaluation reports
-> storage repositories
-> FastAPI service
```

## Recommended Dataset

The primary dataset will be Amazon Reviews 2023, starting with one category such as `Amazon_Fashion`, `Electronics`, or `Beauty_and_Personal_Care`.

The dataset is useful because it includes user-item interactions, ratings, timestamps, review text, item metadata, prices, categories, and product descriptions.

Dataset file formats are documented in [docs/datasets.md](docs/datasets.md).

For local scale testing without downloading external data, CommerceMind also
includes a deterministic synthetic dataset generator documented in
[docs/synthetic_dataset.md](docs/synthetic_dataset.md).

## Evaluation Metrics

- Retrieval quality: `Recall@K`, `HitRate@K`, `MRR`
- Ranking quality: `Precision@K`, `Recall@K`, `HitRate@K`, `MRR@K`
- System quality: `p50` latency, `p95` latency, throughput, index size
- Monitoring: request count, latency, error rate, and empty-result rate

## Completed Roadmap

1. Project definition and repository structure
2. Data ingestion and cleaning
3. Baseline retrieval models
4. Semantic retrieval with vector search
5. Hybrid retrieval with rank fusion
6. Feature-based ranking
7. Personalized recommendations
8. Database persistence layer
9. Offline experiment reporting
10. Real dataset integration
11. Persistent FAISS vector indexes
12. Learning-to-rank model training
13. Docker and deployment documentation
14. Runtime monitoring
15. GitHub Actions CI
16. Performance benchmarking and system quality reports
17. Synthetic scale dataset generation and processed-dataset benchmarking
18. Processed-dataset recommendation accuracy evaluation with chronological holdout

## Current Status

- FastAPI service with health, search, and recommendation endpoints
- Typed request and response schemas
- Dataset path management, ingestion, normalization, and materialization
- Lexical, configurable Sentence Transformer vector, hybrid, and popularity retrieval
- Feature-based ranking over retrieved candidates
- Lightweight learned ranking model training with saved JSON artifacts
- Personalized recommendations from interaction history
- SQLAlchemy storage models and repositories for products and interactions
- Configurable catalog loading from sample data or a seeded database
- Persistent FAISS vector indexes with embedding and catalog compatibility checks
- Deterministic synthetic dataset generator for larger benchmark catalogs
- Dockerfile and Docker Compose configuration for local deployment demos
- Runtime request, latency, error, and empty-result metrics
- GitHub Actions CI for linting, MyPy, coverage, packaging, evaluation, and artifact smoke checks
- In-process performance benchmarking for latency, throughput, error rate, and artifact size
- Retrieval, recommendation, and processed-dataset holdout metrics with offline experiment reports
- Test coverage for API, data pipeline, retrieval, ranking, recommendations, storage, and evaluation modules

## Documentation

- [Project showcase](docs/project_showcase.md)
- [Results snapshot](docs/results.md)
- [Dataset formats](docs/datasets.md)
- [Synthetic dataset generation](docs/synthetic_dataset.md)
- [Performance benchmarking](docs/performance.md)
- [Monitoring](docs/monitoring.md)
- [Deployment](docs/deployment.md)
- [Continuous integration](docs/ci.md)

## Development

Create and activate the virtual environment:

```powershell
uv venv --python 3.12 .venv
.\.venv\Scripts\activate
uv pip install -e ".[dev]"
```

Run tests:

```powershell
python -m pytest
```

Run linting:

```powershell
ruff check src tests
```

Run static type checking:

```powershell
mypy src
```

CI details are documented in [docs/ci.md](docs/ci.md).

Generate offline experiment reports:

```powershell
python -m commercemind.evaluation.run_experiment --benchmark sample --k 2
```

This writes:

```text
reports/offline_experiment.json
reports/offline_experiment.md
```

Available local benchmarks:

- `demo`: tiny smoke-test catalog used for quick checks.
- `sample`: larger deterministic benchmark with products, interactions, search labels, and held-out recommendation labels.

Generate a deterministic synthetic scale dataset:

```powershell
python -m commercemind.data.generate_synthetic_dataset --dataset synthetic_large --products 1000 --users 100 --interactions 5000 --seed 42
```

This writes processed Parquet files under `data/processed/synthetic_large/`.
That directory is ignored by git, so the dataset is regenerated locally instead
of committed.

Generate performance benchmark reports:

```powershell
python -m commercemind.evaluation.run_performance_benchmark --benchmark sample --requests 100 --top-k 5 --output-dir reports/performance
```

Run performance benchmarks against a processed dataset:

```powershell
python -m commercemind.evaluation.run_performance_benchmark --dataset synthetic_large --requests 25 --top-k 5 --output-dir reports/performance-synthetic
```

Evaluate recommendation accuracy on a processed dataset:

```powershell
python -m commercemind.evaluation.run_experiment --dataset synthetic_large --k 5 --output-dir reports/recommendation-synthetic
```

Performance details are documented in [docs/performance.md](docs/performance.md).

Seed a database from normalized dataset files:

```powershell
python -m commercemind.storage.load_dataset --dataset amazon_fashion --database-url sqlite+pysqlite:///./work/commercemind.db
```

Choose the catalog source:

```powershell
$env:COMMERCE_MIND_CATALOG_SOURCE="sample"
```

Use `sample` for the built-in demo catalog. Use `database` after seeding a SQLite or Postgres database:

```powershell
$env:COMMERCE_MIND_CATALOG_SOURCE="database"
$env:COMMERCE_MIND_DATABASE_URL="sqlite+pysqlite:///./work/commercemind.db"
```

Build a persistent FAISS vector index:

```powershell
python -m commercemind.retrieval.build_index --source sample --output-dir artifacts/indexes/products
```

Use the saved vector index when serving search:

```powershell
$env:COMMERCE_MIND_VECTOR_INDEX_DIR="artifacts/indexes/products"
```

Train a learned ranking model:

```powershell
python -m commercemind.ranking.training --benchmark sample --output-path artifacts/models/ranker.json
```

Use the learned ranker when serving search:

```powershell
$env:COMMERCE_MIND_RANKER_MODEL_PATH="artifacts/models/ranker.json"
```

Run the API:

```powershell
python -m uvicorn commercemind.main:app --reload
```

Run with Docker Compose:

```powershell
docker compose up --build api
```

Deployment details are documented in [docs/deployment.md](docs/deployment.md).

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

Read runtime metrics:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

Monitoring details are documented in [docs/monitoring.md](docs/monitoring.md).

Example requests:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search -ContentType "application/json" -Body '{"query":"running shoes","top_k":5}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/recommendations -ContentType "application/json" -Body '{"user_id":"user-runner","top_k":5}'
```

If the local `.venv` launcher points to a missing Python installation, recreate the environment:

```powershell
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
```

# CommerceMind Project Showcase

CommerceMind is a production-style product discovery backend for e-commerce
catalogs. It combines search, recommendations, ranking, learned ranking,
evaluation, persistence, deterministic synthetic scale data, observability,
benchmarking, Docker packaging, and CI.

## One-Line Summary

Built a retrieval and ranking system for e-commerce product discovery with
hybrid lexical/vector search, personalized recommendations, learned ranking,
FAISS indexing, offline evaluation, synthetic scale benchmarking, runtime
monitoring, performance benchmarks, Docker deployment, and GitHub Actions CI.

## System Architecture

```text
Raw, sample, or synthetic dataset
-> ingestion and normalization
-> products and interactions tables
-> lexical retrieval + vector retrieval
-> hybrid candidate generation
-> feature extraction
-> linear or learned ranker
-> search and recommendation APIs
-> offline evaluation reports
-> runtime metrics and performance reports
```

## Core Capabilities

- Natural language product search with lexical, Sentence Transformer vector, and hybrid retrieval.
- Personalized recommendations from user interaction history.
- Feature-based ranking using title, category, brand, description, price intent, and retrieval score.
- Learned ranker training from labeled benchmark queries.
- Persistent FAISS indexes with embedding metadata and catalog-fingerprint validation.
- Deterministic synthetic catalog generation for larger processed-dataset benchmarks.
- SQLite/Postgres-compatible storage through SQLAlchemy repositories.
- Offline quality evaluation with Precision@K, Recall@K, HitRate@K, and MRR@K.
- Chronological recommendation holdout evaluation on processed datasets.
- Runtime monitoring through `/metrics`.
- In-process latency and throughput benchmarks.
- Docker and Docker Compose deployment flow.
- GitHub Actions CI for linting, tests, evaluation, benchmark, ranker, index, and compose validation.

## Resume Bullets

- Built an end-to-end e-commerce discovery backend using FastAPI, Polars, SQLAlchemy, Sentence Transformers, FAISS, and PostgreSQL/SQLite.
- Implemented hybrid lexical/vector retrieval with Reciprocal Rank Fusion, explainable ranking features, and a learned linear ranker trained on separate labeled queries.
- Validated search and recommendation quality with reproducible offline reports, chronological holdouts, 97 tests, 91% coverage, MyPy, Ruff, and GitHub Actions CI.
- Benchmarked a deterministic 1,000-product/5,000-interaction workload and packaged the service with Docker plus versioned, catalog-validated model/index artifacts.

## Interview Talking Points

### Retrieval

CommerceMind first retrieves candidate products instead of ranking the entire
catalog. Lexical retrieval is strong for exact keywords, while vector retrieval
helps with semantic matches. Hybrid retrieval combines both so the system has
better recall than either one alone.

### Ranking

Retrieval finds candidates; ranking decides final order. The ranker uses features
such as retrieval score, title overlap, category overlap, brand overlap,
description overlap, and price-intent match. The learned ranker trains weights
for those features from labeled benchmark queries.

### Evaluation

The system measures quality with Precision@K, Recall@K, HitRate@K, and MRR@K.
This proves ranking quality with repeatable experiments instead of relying on
manual inspection.

Learned-ranker training queries are separated from the queries used to report
search metrics, preventing direct train/evaluation leakage in the benchmark.

For processed datasets, the recommendation evaluator trains on earlier
interactions and hides a future interaction window as ground truth. This tests
whether the recommender can recover items the user interacted with later.

### Scale Testing

Small fixtures keep tests fast, while the synthetic dataset generator creates a
larger processed catalog for local benchmarks. This makes the project easier to
review because anyone can regenerate the same 1000-product workload without
downloading external files.

### Production Readiness

The project includes persistence, Docker deployment, CI, monitoring, and
performance benchmarking. That makes it closer to a real backend service than a
single notebook or isolated ML script.

## Demo Flow

1. Run tests and linting.
2. Generate offline evaluation reports.
3. Train the learned ranker.
4. Build the FAISS vector index.
5. Generate the synthetic scale dataset.
6. Run performance benchmarks.
7. Start the API.
8. Call `/search`, `/recommendations`, and `/metrics`.

## Strongest Technical Claims

- The service separates candidate generation from ranking.
- Retrieval sources are swappable through common protocols.
- Ranking features are explainable and reusable for both hand-written and learned rankers.
- FAISS indexes and learned rankers are saved as artifacts.
- Quality and performance are measured with reproducible reports on both small fixtures and a generated larger catalog.
- The project has automated tests and CI gates across backend, data, ML, evaluation, monitoring, and deployment config.

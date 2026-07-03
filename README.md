# CommerceMind

CommerceMind is a product discovery service for e-commerce catalogs. It combines data ingestion, baseline retrieval, offline evaluation, and API serving as the foundation for semantic search and personalized recommendations.

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
-> supervised ranking
-> re-ranking rules
-> FastAPI service
-> dashboard and monitoring
```

## Recommended Dataset

The primary dataset will be Amazon Reviews 2023, starting with one category such as `Amazon_Fashion`, `Electronics`, or `Beauty_and_Personal_Care`.

The dataset is useful because it includes user-item interactions, ratings, timestamps, review text, item metadata, prices, categories, and product descriptions.

## Evaluation Metrics

- Retrieval quality: `Recall@K`, `HitRate@K`, `MRR`
- Ranking quality: `NDCG@K`, `MAP@K`, `Precision@K`
- Catalog quality: coverage, novelty, diversity
- System quality: `p50` latency, `p95` latency, throughput, index size
- Monitoring: empty-result rate, category drift, embedding drift, popularity drift

## Roadmap

1. Project definition and repository structure
2. Data ingestion and cleaning
3. Baseline retrieval models
4. Semantic retrieval with vector search
5. Two-tower retrieval model
6. Supervised ranking model
7. Diversity-aware re-ranking
8. FastAPI serving layer
9. Experiment tracking and reproducible pipelines
10. Monitoring and drift checks
11. Dashboard and operational views
12. Deployment documentation

## Current Status

- FastAPI service with health and search endpoints
- Typed request and response schemas
- Dataset path management, ingestion, normalization, and materialization
- Lexical and popularity retrieval baselines
- Retrieval metrics and evaluation reports
- Test coverage for API, data pipeline, retrieval, and evaluation modules

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

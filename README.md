# CommerceMind

CommerceMind is a product discovery service for e-commerce catalogs. It combines data ingestion, lexical and semantic retrieval, ranking, personalized recommendations, offline evaluation, database persistence, and API serving.

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
5. Hybrid retrieval with rank fusion
6. Feature-based ranking
7. Personalized recommendations
8. Database persistence layer
9. Offline experiment reporting
10. Real dataset integration
11. Persistent FAISS vector indexes
12. Learning-to-rank model training
13. Docker and deployment documentation

## Current Status

- FastAPI service with health, search, and recommendation endpoints
- Typed request and response schemas
- Dataset path management, ingestion, normalization, and materialization
- Lexical, semantic vector, hybrid, and popularity retrieval
- Feature-based ranking over retrieved candidates
- Personalized recommendations from interaction history
- SQLAlchemy storage models and repositories for products and interactions
- Retrieval and recommendation metrics with offline experiment reports
- Test coverage for API, data pipeline, retrieval, ranking, recommendations, storage, and evaluation modules

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

Run the API:

```powershell
python -m uvicorn commercemind.main:app --reload
```

Open the interactive API docs:

```text
http://127.0.0.1:8000/docs
```

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

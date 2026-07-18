# Deployment

CommerceMind can run as a local Python process or as a Dockerized API service.
The Docker path is useful for demos because it packages the FastAPI server, project
code, runtime dependencies, healthcheck, mounted artifacts, and optional database
support behind one command.

## Build The Image

```powershell
docker build -t commercemind:local .
```

## Run With Docker Compose

```powershell
docker compose up --build api
```

The API listens on:

```text
http://127.0.0.1:8000
```

Check health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Check runtime metrics:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

Search:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search -ContentType "application/json" -Body '{"query":"running shoes","top_k":5}'
```

Recommendations:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/recommendations -ContentType "application/json" -Body '{"user_id":"user-runner","top_k":5}'
```

## Runtime Data

The compose service mounts these local folders:

```text
./work -> /app/work
./artifacts -> /app/artifacts
./reports -> /app/reports
```

Use `work` for local SQLite databases, `artifacts` for FAISS indexes and learned
ranker models, and `reports` for offline evaluation outputs.

## Optional Artifacts

Build a FAISS index:

```powershell
python -m commercemind.retrieval.build_index --source sample --output-dir artifacts/indexes/products
```

For transformer-based semantic retrieval, build and serve with the same model:

```powershell
python -m commercemind.retrieval.build_index --source sample --embedding-backend sentence_transformer --embedding-model sentence-transformers/all-MiniLM-L6-v2 --output-dir artifacts/indexes/products
$env:COMMERCE_MIND_EMBEDDING_BACKEND="sentence_transformer"
$env:COMMERCE_MIND_EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
```

Index artifacts include the embedding identity and a catalog fingerprint. The API
refuses to start with an index built from a different catalog or embedding backend.

Train a learned ranker:

```powershell
python -m commercemind.ranking.training --benchmark sample --output-path artifacts/models/ranker.json
```

Serve with those artifacts:

```powershell
docker compose run --rm -e COMMERCE_MIND_VECTOR_INDEX_DIR=/app/artifacts/indexes/products -e COMMERCE_MIND_RANKER_MODEL_PATH=/app/artifacts/models/ranker.json --service-ports api
```

## Optional Postgres

Start Postgres with the database profile:

```powershell
docker compose --profile database up -d postgres
```

Use this database URL for database-backed mode:

```text
postgresql+psycopg://postgres:postgres@postgres:5432/commercemind
```

When running the API in database mode, seed the database first from processed
dataset files, then set:

```powershell
$env:COMMERCE_MIND_CATALOG_SOURCE="database"
$env:COMMERCE_MIND_DATABASE_URL="postgresql+psycopg://postgres:postgres@postgres:5432/commercemind"
```

## Production Notes

For a real deployment, use a managed Postgres database, upload FAISS/model
artifacts through CI/CD, keep secrets outside the image, expose only the API port,
and run offline evaluation before promoting a new index or ranker.

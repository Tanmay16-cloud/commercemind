# Continuous Integration

CommerceMind uses GitHub Actions to keep the project healthy on pushes and pull
requests. The workflow lives at:

```text
.github/workflows/ci.yml
```

## What CI Checks

The `quality` job runs on Ubuntu with Python 3.12 and performs these gates:

- Installs the project with development dependencies.
- Runs Ruff linting over `src` and `tests`.
- Runs MyPy over the application package.
- Runs the full pytest suite with a 90% coverage gate.
- Builds the distributable wheel.
- Generates a demo offline evaluation report.
- Generates a demo performance benchmark report.
- Generates a small synthetic processed dataset.
- Generates a synthetic recommendation holdout report.
- Generates a synthetic processed-dataset performance report.
- Trains a demo learned ranker artifact.
- Builds a demo FAISS vector index.
- Validates the Docker Compose configuration.

## Why These Gates Matter

Linting catches style and import issues before review.

Tests catch regressions across retrieval, ranking, recommendations, storage, API
serving, monitoring, and deployment config.

The experiment, ranker, and vector-index smoke checks prove that the project is
not only importable, but also able to run its core ML/search workflows end to end.

Docker Compose validation catches deployment configuration mistakes without
requiring the CI job to build a full container image.

## Local Equivalent

Run the closest local equivalent before pushing:

```powershell
ruff check src tests
mypy src
python -m pytest --cov=src/commercemind --cov-fail-under=90
python -m commercemind.evaluation.run_experiment --benchmark demo --k 2 --output-dir reports/ci
python -m commercemind.evaluation.run_performance_benchmark --benchmark demo --requests 5 --top-k 2 --output-dir reports/ci-performance
python -m commercemind.data.generate_synthetic_dataset --dataset synthetic_ci --products 50 --users 10 --interactions 200 --seed 7
python -m commercemind.evaluation.run_experiment --dataset synthetic_ci --k 5 --output-dir reports/ci-recommendation-synthetic
python -m commercemind.evaluation.run_performance_benchmark --dataset synthetic_ci --requests 5 --top-k 2 --output-dir reports/ci-performance-synthetic
python -m commercemind.ranking.training --benchmark demo --output-path artifacts/models/ci-ranker.json
python -m commercemind.retrieval.build_index --source sample --output-dir artifacts/indexes/ci-products
```

If Docker is installed, also run:

```powershell
docker compose config --quiet
```

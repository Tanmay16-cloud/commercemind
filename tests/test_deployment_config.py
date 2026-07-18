from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_defines_api_runtime() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in dockerfile
    assert "python -m pip install --no-cache-dir ." in dockerfile
    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "commercemind.main:app" in dockerfile
    assert '"0.0.0.0"' in dockerfile


def test_dockerignore_excludes_generated_artifacts() -> None:
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    for ignored_path in [
        ".venv/",
        "data/raw/",
        "data/processed/",
        "artifacts/",
        "reports/",
        "work/",
        "*.faiss",
        ".env",
    ]:
        assert ignored_path in dockerignore


def test_compose_file_exposes_api_and_runtime_volumes() -> None:
    compose = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "api:" in compose
    assert "postgres:" in compose
    assert '"8000:8000"' in compose
    assert "COMMERCE_MIND_CATALOG_SOURCE: sample" in compose
    assert "./work:/app/work" in compose
    assert "./artifacts:/app/artifacts" in compose
    assert "./reports:/app/reports" in compose
    assert "healthcheck:" in compose
    assert "profiles:" in compose


def test_deployment_docs_include_demo_commands() -> None:
    docs = (PROJECT_ROOT / "docs" / "deployment.md").read_text(encoding="utf-8")

    assert "docker build -t commercemind:local ." in docs
    assert "docker compose up --build api" in docs
    assert "Invoke-RestMethod http://127.0.0.1:8000/health" in docs
    assert "COMMERCE_MIND_VECTOR_INDEX_DIR" in docs
    assert "COMMERCE_MIND_RANKER_MODEL_PATH" in docs

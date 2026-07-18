from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_prefix="COMMERCE_MIND_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CommerceMind"
    environment: str = "development"
    log_level: str = "INFO"

    project_root: Path = Field(default_factory=_default_project_root)
    data_dir: Path | None = None
    raw_data_dir: Path | None = None
    processed_data_dir: Path | None = None
    artifacts_dir: Path | None = None
    models_dir: Path | None = None
    indexes_dir: Path | None = None
    vector_index_dir: Path | None = None
    ranker_model_path: Path | None = None
    reports_dir: Path | None = None

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/commercemind"
    catalog_source: str = "sample"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: Literal["hashing", "sentence_transformer"] = "hashing"
    retrieval_top_k: int = 100
    ranking_top_k: int = 20

    def model_post_init(self, __context: object) -> None:
        data_dir = self.data_dir or self.project_root / "data"
        self.data_dir = data_dir
        self.raw_data_dir = self.raw_data_dir or data_dir / "raw"
        self.processed_data_dir = self.processed_data_dir or data_dir / "processed"
        self.artifacts_dir = self.artifacts_dir or self.project_root / "artifacts"
        self.models_dir = self.models_dir or self.artifacts_dir / "models"
        self.indexes_dir = self.indexes_dir or self.artifacts_dir / "indexes"
        self.reports_dir = self.reports_dir or self.project_root / "reports"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

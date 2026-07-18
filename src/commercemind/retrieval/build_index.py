from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from commercemind.config import get_settings
from commercemind.retrieval.embeddings import create_text_embedder
from commercemind.retrieval.vector_index import (
    VectorIndexError,
    build_faiss_vector_index,
)
from commercemind.services.catalog import CatalogLoadError, load_catalog_data


@dataclass(frozen=True)
class VectorIndexBuildResult:
    source: str
    output_dir: Path
    vector_count: int
    dimensions: int


def build_vector_index_from_catalog(
    source: str | None = None,
    *,
    database_url: str | None = None,
    output_dir: Path | None = None,
    embedding_backend: str | None = None,
    embedding_model_name: str | None = None,
) -> VectorIndexBuildResult:
    settings = get_settings()
    catalog_source = source or settings.catalog_source
    target_dir = output_dir or _default_output_dir()

    catalog = load_catalog_data(catalog_source, database_url or settings.database_url)
    backend = embedding_backend or settings.embedding_backend
    model_name = embedding_model_name or settings.embedding_model_name
    index = build_faiss_vector_index(
        catalog.products,
        create_text_embedder(backend, model_name=model_name),
    )
    index.save(target_dir)

    return VectorIndexBuildResult(
        source=catalog_source,
        output_dir=target_dir,
        vector_count=index.size,
        dimensions=index.dimensions,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a persistent FAISS vector index for product semantic search."
    )
    parser.add_argument(
        "--source",
        choices=["sample", "database"],
        default=None,
        help="Catalog source to index. Defaults to COMMERCE_MIND_CATALOG_SOURCE.",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy database URL when --source database is used.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["hashing", "sentence_transformer"],
        default=None,
        help="Embedding backend. Defaults to COMMERCE_MIND_EMBEDDING_BACKEND.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Sentence Transformer model name when that backend is selected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for products.faiss and documents.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = build_vector_index_from_catalog(
            args.source,
            database_url=args.database_url,
            output_dir=args.output_dir,
            embedding_backend=args.embedding_backend,
            embedding_model_name=args.embedding_model,
        )
    except (CatalogLoadError, VectorIndexError) as exc:
        parser.error(str(exc))

    print(
        "Built FAISS vector index "
        f"from '{result.source}' catalog with "
        f"{result.vector_count} vectors, "
        f"{result.dimensions} dimensions, "
        f"at {result.output_dir}."
    )
    return 0


def _default_output_dir() -> Path:
    settings = get_settings()
    if settings.indexes_dir is None:
        raise VectorIndexError("indexes_dir is not configured")
    return settings.indexes_dir / "products"


if __name__ == "__main__":
    raise SystemExit(main())

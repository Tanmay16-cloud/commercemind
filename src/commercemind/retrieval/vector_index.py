from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import faiss
import numpy as np
import polars as pl

from commercemind.retrieval.embeddings import (
    TextEmbedder,
    embedder_identity,
    l2_normalize,
    weighted_text,
)

INDEX_FILENAME = "products.faiss"
METADATA_FILENAME = "documents.json"
ARTIFACT_VERSION = 2


@dataclass(frozen=True)
class VectorDocument:
    item_id: str
    title: str
    text: str


@dataclass(frozen=True)
class VectorIndexSearchResult:
    document: VectorDocument
    score: float


class VectorIndexError(RuntimeError):
    pass


class FaissVectorIndex:
    def __init__(
        self,
        documents: list[VectorDocument],
        index: faiss.Index,
        *,
        embedding_backend: str,
        embedding_model: str | None,
        catalog_fingerprint: str,
    ) -> None:
        if index.ntotal != len(documents):
            raise VectorIndexError("index vector count does not match document count")

        self._documents = documents
        self._index = index
        self._embedding_backend = embedding_backend
        self._embedding_model = embedding_model
        self._catalog_fingerprint = catalog_fingerprint

    @property
    def documents(self) -> list[VectorDocument]:
        return list(self._documents)

    @property
    def dimensions(self) -> int:
        return int(self._index.d)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    @property
    def embedding_backend(self) -> str:
        return self._embedding_backend

    @property
    def embedding_model(self) -> str | None:
        return self._embedding_model

    @property
    def catalog_fingerprint(self) -> str:
        return self._catalog_fingerprint

    @classmethod
    def build(
        cls,
        documents: list[VectorDocument],
        vectors: np.ndarray,
        *,
        embedding_backend: str,
        embedding_model: str | None,
        catalog_fingerprint: str,
    ) -> FaissVectorIndex:
        normalized_vectors = _prepare_vectors(vectors)
        index = faiss.IndexFlatIP(normalized_vectors.shape[1])
        index.add(normalized_vectors)
        return cls(
            documents=documents,
            index=index,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            catalog_fingerprint=catalog_fingerprint,
        )

    def validate_catalog(self, products: pl.DataFrame) -> None:
        current_fingerprint = catalog_fingerprint(products)
        if current_fingerprint != self.catalog_fingerprint:
            raise VectorIndexError(
                "vector index catalog fingerprint does not match the active catalog; "
                "rebuild the index before serving requests"
            )

    def search(self, query_vector: np.ndarray, top_k: int) -> list[VectorIndexSearchResult]:
        if top_k <= 0 or self.size == 0:
            return []

        prepared_query = _prepare_query_vector(query_vector, self.dimensions)
        search_k = min(top_k, self.size)
        scores, indexes = self._index.search(prepared_query.reshape(1, -1), search_k)

        results: list[VectorIndexSearchResult] = []
        for score, index in zip(scores[0], indexes[0], strict=True):
            if index < 0 or score <= 0:
                continue
            results.append(
                VectorIndexSearchResult(
                    document=self._documents[int(index)],
                    score=float(score),
                )
            )

        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / INDEX_FILENAME))
        (directory / METADATA_FILENAME).write_text(
            json.dumps(_metadata_payload(self), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> FaissVectorIndex:
        index_path = directory / INDEX_FILENAME
        metadata_path = directory / METADATA_FILENAME

        if not index_path.exists() or not metadata_path.exists():
            raise VectorIndexError(f"missing vector index artifact in {directory}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("version") != ARTIFACT_VERSION:
            raise VectorIndexError("unsupported vector index artifact version")

        documents = [
            VectorDocument(
                item_id=str(document["item_id"]),
                title=str(document["title"]),
                text=str(document["text"]),
            )
            for document in metadata["documents"]
        ]
        index = faiss.read_index(str(index_path))

        if int(metadata["dimensions"]) != int(index.d):
            raise VectorIndexError("metadata dimensions do not match FAISS index dimensions")

        if int(metadata["document_count"]) != len(documents):
            raise VectorIndexError("metadata document count does not match stored documents")

        return cls(
            documents=documents,
            index=index,
            embedding_backend=str(metadata["embedding_backend"]),
            embedding_model=(
                str(metadata["embedding_model"])
                if metadata.get("embedding_model") is not None
                else None
            ),
            catalog_fingerprint=str(metadata["catalog_fingerprint"]),
        )


def build_faiss_vector_index(
    products: pl.DataFrame,
    embedder: TextEmbedder,
) -> FaissVectorIndex:
    documents = build_vector_documents(products)
    vectors = embedder.encode([document.text for document in documents])
    embedding_backend, embedding_model = embedder_identity(embedder)
    return FaissVectorIndex.build(
        documents,
        vectors,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        catalog_fingerprint=catalog_fingerprint(products),
    )


def build_vector_documents(products: pl.DataFrame) -> list[VectorDocument]:
    return [_product_document(row) for row in products.iter_rows(named=True)]


def _metadata_payload(index: FaissVectorIndex) -> dict[str, object]:
    return {
        "version": ARTIFACT_VERSION,
        "dimensions": index.dimensions,
        "document_count": index.size,
        "embedding_backend": index.embedding_backend,
        "embedding_model": index.embedding_model,
        "catalog_fingerprint": index.catalog_fingerprint,
        "documents": [asdict(document) for document in index.documents],
    }


def catalog_fingerprint(products: pl.DataFrame) -> str:
    fields = ("item_id", "title", "category", "brand", "description", "price")
    rows = [
        {field: row.get(field) for field in fields}
        for row in products.sort("item_id").iter_rows(named=True)
    ]
    serialized = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(serialized.encode("utf-8")).hexdigest()


def _prepare_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise VectorIndexError("vectors must be a two-dimensional array")
    if vectors.shape[1] <= 0:
        raise VectorIndexError("vectors must have at least one dimension")

    return np.ascontiguousarray(l2_normalize(vectors).astype(np.float32))


def _prepare_query_vector(query_vector: np.ndarray, dimensions: int) -> np.ndarray:
    if query_vector.ndim != 1:
        raise VectorIndexError("query vector must be one-dimensional")
    if query_vector.shape[0] != dimensions:
        raise VectorIndexError("query vector dimensions do not match index dimensions")

    return np.ascontiguousarray(l2_normalize(query_vector.reshape(1, -1))[0].astype(np.float32))


def _product_document(row: dict[str, object]) -> VectorDocument:
    title = str(row["title"])
    text = weighted_text(
        [
            (_optional_text(row, "title"), 3.0),
            (_optional_text(row, "category"), 2.0),
            (_optional_text(row, "brand"), 1.0),
            (_optional_text(row, "description"), 1.0),
        ]
    )

    return VectorDocument(
        item_id=str(row["item_id"]),
        title=title,
        text=text,
    )


def _optional_text(row: dict[str, object], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    return str(value)

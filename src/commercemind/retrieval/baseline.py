from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import polars as pl


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class RetrievalCandidate:
    item_id: str
    title: str
    score: float


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return TOKEN_PATTERN.findall(text.lower())


class PopularityRetriever:
    def __init__(self, products: pl.DataFrame, interactions: pl.DataFrame) -> None:
        self._titles = _product_titles(products)
        counts = interactions.group_by("item_id").len(name="interaction_count")
        ranked = counts.sort("interaction_count", descending=True)

        self._candidates = [
            RetrievalCandidate(
                item_id=row["item_id"],
                title=self._titles.get(row["item_id"], row["item_id"]),
                score=float(row["interaction_count"]),
            )
            for row in ranked.iter_rows(named=True)
        ]

    def retrieve(self, top_k: int) -> list[RetrievalCandidate]:
        return self._candidates[:top_k]


class LexicalRetriever:
    def __init__(self, products: pl.DataFrame) -> None:
        self._documents = [
            _ProductDocument(
                item_id=row["item_id"],
                title=row["title"],
                tokens=tokenize(_search_text(row)),
            )
            for row in products.iter_rows(named=True)
        ]
        self._document_count = len(self._documents)
        self._idf = self._build_idf(self._documents)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_counts = Counter(query_tokens)
        scored = []

        for document in self._documents:
            document_counts = Counter(document.tokens)
            score = 0.0

            for token, query_count in query_counts.items():
                if token in document_counts:
                    score += query_count * document_counts[token] * self._idf.get(token, 0.0)

            if score > 0:
                scored.append(
                    RetrievalCandidate(
                        item_id=document.item_id,
                        title=document.title,
                        score=score,
                    )
                )

        return sorted(scored, key=lambda candidate: candidate.score, reverse=True)[:top_k]

    def _build_idf(self, documents: Iterable[_ProductDocument]) -> dict[str, float]:
        document_frequency: Counter[str] = Counter()

        for document in documents:
            document_frequency.update(set(document.tokens))

        return {
            token: math.log((1 + self._document_count) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }


@dataclass(frozen=True)
class _ProductDocument:
    item_id: str
    title: str
    tokens: list[str]


def _product_titles(products: pl.DataFrame) -> dict[str, str]:
    return {row["item_id"]: row["title"] for row in products.iter_rows(named=True)}


def _search_text(row: dict[str, object]) -> str:
    fields = [
        row.get("title"),
        row.get("category"),
        row.get("brand"),
        row.get("description"),
    ]
    return " ".join(str(field) for field in fields if field is not None)

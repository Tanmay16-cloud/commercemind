from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from commercemind.retrieval.baseline import RetrievalCandidate


class SearchRetriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        ...


@dataclass(frozen=True)
class RetrieverWeight:
    retriever: SearchRetriever
    weight: float = 1.0


class HybridRetriever:
    def __init__(
        self,
        retrievers: Sequence[SearchRetriever | RetrieverWeight],
        *,
        rrf_k: int = 60,
        candidate_pool_size: int = 50,
    ) -> None:
        if not retrievers:
            raise ValueError("at least one retriever is required")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive")
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")

        self._retrievers = [_as_weighted_retriever(retriever) for retriever in retrievers]
        self._rrf_k = rrf_k
        self._candidate_pool_size = candidate_pool_size

    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        if top_k <= 0:
            return []

        scores: dict[str, float] = {}
        candidates_by_item_id: dict[str, RetrievalCandidate] = {}

        for weighted_retriever in self._retrievers:
            candidates = weighted_retriever.retriever.retrieve(query, self._candidate_pool_size)

            for rank, candidate in enumerate(candidates, start=1):
                scores[candidate.item_id] = scores.get(candidate.item_id, 0.0) + (
                    weighted_retriever.weight / (self._rrf_k + rank)
                )
                candidates_by_item_id.setdefault(candidate.item_id, candidate)

        ranked_item_ids = sorted(
            scores,
            key=lambda item_id: (-scores[item_id], candidates_by_item_id[item_id].title),
        )

        return [
            RetrievalCandidate(
                item_id=item_id,
                title=candidates_by_item_id[item_id].title,
                score=scores[item_id],
            )
            for item_id in ranked_item_ids[:top_k]
        ]


def _as_weighted_retriever(retriever: SearchRetriever | RetrieverWeight) -> RetrieverWeight:
    if isinstance(retriever, RetrieverWeight):
        if retriever.weight <= 0:
            raise ValueError("retriever weight must be positive")
        return retriever

    return RetrieverWeight(retriever=retriever)

"""Hybrid candidate generation through deterministic reciprocal-rank fusion."""

from __future__ import annotations

from collections.abc import Mapping


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, list[tuple[str, float]]],
    *,
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Fuse distinct ranked lists without comparing incomparable raw scores."""
    if k < 1 or top_k < 1:
        raise ValueError("k and top_k must be positive")
    if not ranked_lists:
        raise ValueError("at least one candidate list is required")
    total: dict[str, float] = {}
    for results in ranked_lists.values():
        observed: set[str] = set()
        for rank, (offer_id, _) in enumerate(results, start=1):
            if offer_id in observed:
                continue
            observed.add(offer_id)
            total[offer_id] = total.get(offer_id, 0.0) + 1.0 / (k + rank)
    return sorted(total.items(), key=lambda item: (-item[1], item[0]))[:top_k]


class HybridRetriever:
    """Combine BM25 and dense result ranks, then optionally rerank with ML."""

    def __init__(
        self,
        bm25: object,
        dense: object,
        *,
        candidate_k: int = 50,
        rrf_k: int = 60,
    ) -> None:
        if candidate_k < 1 or rrf_k < 1:
            raise ValueError("candidate_k and rrf_k must be positive")
        self.bm25 = bm25
        self.dense = dense
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: object, top_k: int) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        depth = max(top_k, self.candidate_k)
        return reciprocal_rank_fusion(
            {
                "bm25": self.bm25.search(query, depth),
                "dense": self.dense.search(query, depth),
            },
            k=self.rrf_k,
            top_k=top_k,
        )

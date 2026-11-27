"""Candidate retrieval metrics with explicit incomplete-qrels limitations."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

from productmatch.retrieval.benchmark import RetrievalBenchmark


def evaluate_retrieval(
    benchmark: RetrievalBenchmark,
    retriever: object,
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, object]:
    """Compute mean recovery and latency over one retrieval call per query.

    When qrels are incomplete, the score is known-positive recovery@K,
    not catalog Recall@K. Never claim unlabeled catalog candidates are negative.
    """
    if not ks or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")
    cutoffs = sorted(set(ks))
    maximum = max(cutoffs)
    counts = {k: [] for k in cutoffs}
    hit_rates = {k: [] for k in cutoffs}
    timings: list[float] = []
    for query in benchmark.queries:
        start = time.perf_counter()
        ranking = retriever.search(query, maximum)
        timings.append((time.perf_counter() - start) * 1000)
        ids = [offer_id for offer_id, _ in ranking]
        if len(ids) != len(set(ids)):
            raise ValueError("retriever returned duplicate offer IDs")
        gold = benchmark.qrels[query.offer_id]
        for k in cutoffs:
            found = len(set(ids[:k]) & gold)
            counts[k].append(found / len(gold))
            hit_rates[k].append(float(found > 0))

    recovery = {str(k): round(sum(counts[k]) / len(counts[k]), 6) for k in cutoffs}
    hit_rate = {str(k): round(sum(hit_rates[k]) / len(hit_rates[k]), 6) for k in cutoffs}
    metric_prefix = "recall" if benchmark.complete_judgments else "known_positive_recovery"
    return {
        "qrels_complete": benchmark.complete_judgments,
        "benchmark_source": benchmark.source,
        "queries": len(benchmark.queries),
        "catalog_size": len(benchmark.catalog),
        f"{metric_prefix}_at_k": recovery,
        "hit_rate_at_k": hit_rate,
        "mean_query_latency_ms": round(sum(timings) / len(timings), 3),
        "p95_query_latency_ms": round(
            sorted(timings)[max(0, math.ceil(0.95 * len(timings)) - 1)], 3
        ),
        "latency_note": (
            "Wall-clock local search latency, includes query encoding but excludes "
            "offline catalog indexing. Hardware/model cache affect timings."
        ),
        "judgment_note": (
            "Complete catalog relevance judgments are asserted by dataset authors."
            if benchmark.complete_judgments
            else "Only known positive pair links are judged; the candidate corpus "
            "is NOT exhaustively labeled. These are diagnostic recovery@K values, "
            "not benchmark catalog Recall@K."
        ),
    }

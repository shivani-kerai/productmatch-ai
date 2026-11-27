"""Session 4 offline retrieval evaluation tests: no model downloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from productmatch.retrieval.benchmark import (
    Offer,
    RetrievalBenchmark,
    benchmark_from_curated,
    benchmark_from_wdc_validation,
)
from productmatch.retrieval.bm25 import BM25Retriever
from productmatch.retrieval.dense import DenseRetriever, normalized_matrix
from productmatch.retrieval.evaluation import evaluate_retrieval
from productmatch.retrieval.fusion import HybridRetriever, reciprocal_rank_fusion


@pytest.fixture
def curated() -> RetrievalBenchmark:
    catalog = (
        Offer("a", "Acme shampoo 250ml", "Acme"),
        Offer("b", "Beta camera X200", "Beta"),
        Offer("c", "Acme shampoo 500ml", "Acme"),
        Offer("d", "Garden hose 20m", "Garden"),
    )
    queries = (
        Offer("q1", "Acme shampoo 250 ml", "Acme"),
        Offer("q2", "Beta camera X200", "Beta"),
    )
    return RetrievalBenchmark(
        catalog=catalog,
        queries=queries,
        qrels={"q1": frozenset({"a"}), "q2": frozenset({"b"})},
        complete_judgments=True,
        source="synthetic fully-judged catalog",
    )


def toy_encoder(texts: list[str]) -> np.ndarray:
    """Deterministic dependency-free test encoder, NOT semantic model."""
    vectors = []
    for text in texts:
        value = text.casefold()
        vectors.append(
            [
                float("shampoo" in value),
                float("camera" in value),
                float("garden" in value),
                float("250" in value),
                float("x200" in value),
            ]
        )
    return np.asarray(vectors, dtype=np.float32)


def test_bm25_recovers_correct_top1(curated: RetrievalBenchmark) -> None:
    bm25 = BM25Retriever(curated.catalog)
    assert bm25.search(curated.queries[0], 1)[0][0] == "a"
    assert bm25.search(curated.queries[1], 1)[0][0] == "b"
    report = evaluate_retrieval(curated, bm25, ks=(1, 2))
    assert report["recall_at_k"]["1"] == 1.0
    assert report["queries"] == 2
    assert "known_positive_recovery_at_k" not in report


def test_bm25_excludes_same_offer_id() -> None:
    bm25 = BM25Retriever((Offer("self", "Small kettle"), Offer("x", "Other kettle")))
    assert [item[0] for item in bm25.search(Offer("self", "Small kettle"), 4)] == ["x"]


def test_dense_cosine_and_shape_checks(curated: RetrievalBenchmark) -> None:
    dense = DenseRetriever(curated.catalog, toy_encoder)
    assert dense.search(curated.queries[0], 1)[0][0] == "a"
    assert dense.search(curated.queries[1], 1)[0][0] == "b"
    with pytest.raises(ValueError, match="zero-norm"):
        normalized_matrix(np.array([[0, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="two-dimensional"):
        normalized_matrix(np.array([1, 2], dtype=np.float32))


def test_hybrid_fusion_is_deterministic(curated: RetrievalBenchmark) -> None:
    hybrid = HybridRetriever(
        BM25Retriever(curated.catalog),
        DenseRetriever(curated.catalog, toy_encoder),
        candidate_k=4,
    )
    first = hybrid.search(curated.queries[0], 3)
    assert first == hybrid.search(curated.queries[0], 3)
    assert first[0][0] == "a"
    fusion = reciprocal_rank_fusion(
        {"bm25": [("a", 5.0), ("b", 2.0)], "dense": [("b", 0.9), ("a", 0.8)]},
        k=60,
        top_k=2,
    )
    assert [item[0] for item in fusion] == ["a", "b"]


def test_incomplete_qrels_are_not_called_recall(curated: RetrievalBenchmark) -> None:
    incomplete = RetrievalBenchmark(
        catalog=curated.catalog,
        queries=curated.queries,
        qrels=curated.qrels,
        complete_judgments=False,
        source="WDC diagnostic",
    )
    result = evaluate_retrieval(incomplete, BM25Retriever(incomplete.catalog), ks=(1,))
    assert result["known_positive_recovery_at_k"]["1"] == 1.0
    assert "recall_at_k" not in result


def test_invalid_benchmark_and_ks(curated: RetrievalBenchmark) -> None:
    with pytest.raises(ValueError, match="every query"):
        RetrievalBenchmark(curated.catalog, curated.queries, {}, True, "invalid")
    with pytest.raises(ValueError, match="relevant offers"):
        RetrievalBenchmark(
            curated.catalog,
            curated.queries,
            {"q1": frozenset({"missing"}), "q2": frozenset({"b"})},
            True,
            "invalid",
        )
    with pytest.raises(ValueError, match="positive integers"):
        evaluate_retrieval(curated, BM25Retriever(curated.catalog), ks=(0,))


def test_wdc_validation_safe_only(tmp_path: Path) -> None:
    pairs = [
        {
            "id_left": 1,
            "id_right": 2,
            "title_left": "Acme shampoo 250ml",
            "title_right": "Acme shampoo 250 ml",
            "brand_left": "Acme",
            "brand_right": "Acme",
            "label": 1,
        },
        {
            "id_left": 1,
            "id_right": 3,
            "title_left": "Acme shampoo 250ml",
            "title_right": "Acme shampoo 500ml",
            "brand_left": "Acme",
            "brand_right": "Acme",
            "label": 0,
        },
    ]
    pd.DataFrame(pairs).to_parquet(tmp_path / "validation_safe.parquet", index=False)
    bench = benchmark_from_wdc_validation(tmp_path)
    assert len(bench.catalog) == 2
    assert len(bench.queries) == 1
    assert bench.qrels["1"] == frozenset({"2"})
    assert bench.complete_judgments is False
    result = evaluate_retrieval(bench, BM25Retriever(bench.catalog), ks=(1,))
    assert result["known_positive_recovery_at_k"]["1"] == 1.0


def test_curated_csv_inputs(tmp_path: Path) -> None:
    pd.DataFrame([{"offer_id": "a", "title": "Acme kettle"}]).to_csv(
        tmp_path / "catalog.csv", index=False
    )
    pd.DataFrame([{"offer_id": "q", "title": "Acme kettle"}]).to_csv(
        tmp_path / "queries.csv", index=False
    )
    pd.DataFrame([{"query_id": "q", "offer_id": "a"}]).to_csv(tmp_path / "qrels.csv", index=False)
    bench = benchmark_from_curated(
        tmp_path / "catalog.csv",
        tmp_path / "queries.csv",
        tmp_path / "qrels.csv",
        complete_judgments=True,
    )
    assert bench.complete_judgments is True
    assert bench.qrels["q"] == frozenset({"a"})

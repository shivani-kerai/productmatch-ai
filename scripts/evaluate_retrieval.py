"""Evaluate product retrieval on development-only or independently curated judgments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from productmatch.retrieval.benchmark import (
    benchmark_from_curated,
    benchmark_from_wdc_validation,
)
from productmatch.retrieval.bm25 import BM25Retriever
from productmatch.retrieval.dense import DenseRetriever, sentence_transformer_encoder
from productmatch.retrieval.evaluation import evaluate_retrieval
from productmatch.retrieval.fusion import HybridRetriever
from productmatch.retrieval.rerank import PairwiseReranker, RerankedRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/wdc50_medium"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/session4_retrieval"))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("bm25", "dense", "hybrid", "reranked"),
        default=("bm25",),
    )
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--faiss", action="store_true")
    parser.add_argument(
        "--reranker-artifact",
        type=Path,
        default=Path("artifacts/session3_baselines/logistic_regression.joblib"),
    )
    parser.add_argument("--catalog-csv", type=Path)
    parser.add_argument("--queries-csv", type=Path)
    parser.add_argument("--qrels-csv", type=Path)
    parser.add_argument("--complete-qrels", action="store_true")
    args = parser.parse_args()

    curated = (args.catalog_csv, args.queries_csv, args.qrels_csv)
    if any(curated) and not all(curated):
        parser.error("curated retrieval requires catalog, queries and qrels CSVs")
    if args.complete_qrels and not all(curated):
        parser.error("--complete-qrels requires independently curated qrels")
    if args.candidate_k < 1 or not args.ks or any(k < 1 for k in args.ks):
        parser.error("--candidate-k and every --ks value must be positive")
    bench = (
        benchmark_from_curated(*curated, complete_judgments=args.complete_qrels)
        if all(curated)
        else benchmark_from_wdc_validation(args.data_dir)
    )
    bm25 = BM25Retriever(bench.catalog)
    requested = dict.fromkeys(args.methods)
    needs_dense = any(x in requested for x in ("dense", "hybrid", "reranked"))
    dense = None
    if needs_dense:
        encoder = sentence_transformer_encoder(args.embedding_model)
        dense = DenseRetriever(bench.catalog, encoder, use_faiss=args.faiss)

    results = {}
    for name in requested:
        if name == "bm25":
            retriever = bm25
        elif name == "dense":
            retriever = dense
        else:
            hybrid = HybridRetriever(bm25, dense, candidate_k=args.candidate_k)
            if name == "hybrid":
                retriever = hybrid
            else:
                reranker = PairwiseReranker(str(args.reranker_artifact), bench.catalog)
                retriever = RerankedRetriever(hybrid, reranker, depth=args.candidate_k)
        results[name] = evaluate_retrieval(bench, retriever, ks=args.ks)

    report = {
        "session": 4,
        "source": bench.source,
        "complete_qrels": bench.complete_judgments,
        "embedding_model": args.embedding_model if needs_dense else None,
        "faiss_enabled": bool(needs_dense and args.faiss),
        "candidate_k": args.candidate_k,
        "results": results,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    output = args.report_dir / "retrieval_metrics.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

# Session 4 — Candidate retrieval and ranking

## Objective

Expand pairwise product matching into **candidate generation**. The default
WDC validation-only benchmark intentionally reports known-positive recovery
at K, not complete catalog recall: the WDC pairwise dataset does not provide
exhaustive query-to-catalog judgments.

## Components

- BM25Retriever: Okapi BM25 with deterministic scoring and ID tie-breaks,
  suitable for a small product catalog.
- DenseRetriever: SentenceTransformers embeddings with cosine similarity.
  Uses a NumPy reference search, or optional exact FAISS IndexFlatIP.
- HybridRetriever: reciprocal-rank fusion across BM25 and dense results.
- PairwiseReranker: optional locally trained Session 3 model for reranking
  the retrieved candidate pool; do not load untrusted joblib artifacts.
- evaluate_retrieval: recovery@1/5/10, hit rate and mean/P95 local query
  latency. Corpus embedding/index build time is excluded.
- Curated benchmark loader: accepts catalog.csv, queries.csv and qrels.csv
  with explicit, independently audited completeness status.

## Windows PowerShell

From C:\productmatch-ai after preparing Session 2 and training Session 3:

1. git fetch origin
2. git switch feature/session-4-candidate-retrieval
3. .\.venv\Scripts\python.exe -m pip install -e ".[dev,data,models]"
4. .\.venv\Scripts\python.exe -m pytest -q
5. .\.venv\Scripts\python.exe scripts/evaluate_retrieval.py --methods bm25

To use a public, pre-trained embedding model for dense retrieval:

1. .\.venv\Scripts\python.exe -m pip install -e ".[retrieval]"
2. .\.venv\Scripts\python.exe scripts/evaluate_retrieval.py --methods bm25 dense hybrid

For an optional exact FAISS backend install .[faiss] and add --faiss.
For supervised reranking, run Session 3 locally first and add
--methods reranked; this loads the trusted model artifact under artifacts/.

The report is stored in reports/session4_retrieval/retrieval_metrics.json,
which is gitignored.

## Independent retrieval dataset

Provide three UTF-8 CSV files:
- catalog.csv: offer_id,title,brand,description,price,currency
- queries.csv: offer_id,title,brand,description,price,currency
- qrels.csv: query_id,offer_id

A qrels row identifies one known relevant catalog offer for a query.
Supply the three paths using --catalog-csv, --queries-csv, --qrels-csv.
Only set --complete-qrels when every possible query/catalog match has
been independently judged under a documented annotation protocol.

Without complete judgments, results are known-positive recovery, not
catalog Recall@K; unknown candidates must not be treated as negatives.

## Benchmark and implementation caveats

- The default candidate catalog consists of right-side offers from
  validation_safe pairs and a positive-pair query on the left side.
- This is a synthetic diagnostic for known-positive recovery. It is not an
  official WDC retrieval protocol, and its scores must not be compared
  to WDC pairwise F1.
- No official WDC seen/unseen test Parquet files are loaded.
- The default sentence-transformer model may download weights at runtime.
  Record its actual model version and comply with model license.
- FAISS IndexFlatIP is **exact**, not approximate nearest neighbor (ANN).
- Local timing is hardware-dependent and includes query embedding but
  excludes building the corpus index.
- Only load local trusted joblib model artifacts. joblib/pickle formats can
  execute code when loaded.

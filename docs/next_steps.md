# Next milestone — Session 5: service and human review

Session 4 adds BM25 and dense candidate generation, rank fusion,
optional supervised reranking, and development-only retrieval diagnostics.

Next:
1. Inspect local Session 4 candidate retrieval results for 1/5/10 cutoff
   recovery and latency.
2. Build an independent, exhaustively labeled query/catalog benchmark to
   measure actual catalog Recall@K; WDC pairwise labels are incomplete.
3. Build FastAPI endpoints for candidate search and pairwise scoring.
4. Create a Streamlit review interface for high-confidence errors and uncertain matches.
5. Save model and dataset provenance; add Docker, API tests and deployment.
6. Keep official WDC test splits sealed until the final selected pipeline.

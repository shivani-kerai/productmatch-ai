# Next milestone — Session 4: retrieval + ranking

Session 3 adds explicit pairwise feature engineering, logistic regression and
LightGBM, development validation threshold tuning, AP/F1, hard-negative error
analysis, saved artifacts and reproducibility metadata.

Next:
1. Run Session 3 locally on the full WDC train and safe validation sets.
2. Inspect high-confidence false positives for size, pack count and model variants.
3. Add calibrated probability metrics with proper held-out calibration splits.
4. Construct a **separate labeled retrieval evaluation dataset** (catalog/query
   design); pairwise pairs are insufficient to claim catalog Recall@K.
5. Benchmark candidate generation (BM25 and embeddings) with Recall@K, latency,
   and realistic catalog cardinality.
6. Preserve unseen/seen official test files until final evaluation.

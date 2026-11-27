# Next milestone — Session 7: model validation and portfolio-ready demonstration

Session 6 implements score calibration diagnostics, SHA-256-bound local
calibration, authenticated request monitoring and deployment safeguards.

Next:
1. Collect **independent, exhaustively judged** retrieval queries to report
   meaningful catalog Recall@K instead of known-positive recovery.
2. Create fresh label data untouched by model selection, then evaluate Brier
   score, ECE, PR-AUC, F1 and confidence bands without tuning on this holdout.
3. Add structured deployment logging, managed secrets and OAuth/RBAC.
4. Benchmark P95 latency under load and record corpus indexing time.
5. Provision private infrastructure and explicitly approve any deployment.
6. Produce a project demo, architectural diagram and truthful benchmark report.

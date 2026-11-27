# Next milestone — Session 8: evidence collection and deployment validation

Session 7 includes a frozen external pairwise evaluation workflow, basic ID
leakage audits, honest portfolio reporting and a local-only API smoke utility.

Before citing real model quality or deploying:
1. Gather independently annotated product pairs whose offer identities do not
   appear in train or validation_safe. Audit annotation quality and source.
2. Curate a fully judged query/catalog relevance set for true catalog Recall@K.
3. Run the frozen model on independent pairs and document confidence intervals,
   subgroups, and the different types of product variant error.
4. Record BM25/dense/hybrid query latency on actual hardware and catalog size.
5. Run an approved local container and smoke test; do not imply public hosting.
6. Perform load, security, dependency and data governance checks before any
   private/cloud deployment.
7. Publish *verified* benchmark metrics and a recorded synthetic-only demo
   with clear data licensing and no secrets.

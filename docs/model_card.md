# Model card — not yet trained

**Status:** Session 1. There is no fitted model or product-match decision API.

The `features` CLI produces normalization and Jaccard *features*, not model probabilities.
No accuracy, F1, recall, precision or production-readiness claims are made.

When models are trained, document:

1. Task definition and intended use / out-of-scope use.
2. Training data, benchmark variant, label semantics and exact splits.
3. Baseline, feature engineering and model hyperparameters.
4. Validation-based threshold selection and calibration method.
5. Held-out results and confidence intervals where suitable.
6. Performance on hard negatives, unseen products and missing attributes.
7. Latency, throughput, memory and monitoring plan.
8. Known limitations, error analysis and version information.

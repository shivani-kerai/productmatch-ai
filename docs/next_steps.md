# Next milestone — Session 3: Baseline product matching

Session 2 introduces the WDC download helper, benchmark-aware schema validation, source split selection, deterministic Parquet preparation, leakage audit, EDA script/notebook, and fixture tests.

Once you download the official archive locally, verify the printed member list and manifest; compare row counts and inspect the EDA reports. Please do **not** publish the raw data without checking its terms.

Then implement:

1. Feature extraction on official train/validation split (token Jaccard, character similarity, brand/model/quantity checks).
2. Transparent rule baseline and logistic regression; log model parameters and train-only feature transforms.
3. Precision, recall, F1 and PR-AUC on validation, with simple confidence intervals.
4. Hard-negative error analysis; avoid turning entity-cluster identifiers into predictive features.
5. Save fitted model, feature schema, and evaluation report with versioned metadata.
6. Keep the official seen/unseen test splits sealed until final comparisons.

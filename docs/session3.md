# Session 3 — Supervised pairwise matching baselines

**Status:** Implementation on `feature/session-3-matching-baselines`.

## Objective

Evaluate whether hand-built, interpretable product attributes can distinguish
**same sellable item** versus nonmatches, including variant and package-size
differences. This is a pairwise classifier; it is **not candidate retrieval**.

## Inputs

- `data/processed/wdc50_medium/train.parquet` — 6,000 official train pairs.
- `data/processed/wdc50_medium/validation_safe.parquet` — 3,473 validation
  pairs after quarantining 27 pairs involving four source-level test-overlap IDs.
- `manifest.json` — provenance and disjoint-offer audit. Missing or inconsistent
  manifests cause a hard failure.
- No `test_seen.parquet`, `test_unseen.parquet`, original
  `validation.parquet`, or test label files are opened while training.

## How to run on Windows PowerShell

```powershell
cd C:\productmatch-ai
git fetch origin
git switch feature/session-3-matching-baselines
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data,models]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/train_baselines.py
```

Train a single model:
```powershell
.\.venv\Scripts\python.exe scripts/train_baselines.py --models logistic_regression
```

## Features (23 fixed numeric values)

Title token Jaccard and containment, normalized exact title, character similarity,
token counts; brand agreement/conflict/missingness; description token overlap;
numeric tokens and alphanumeric model-token differences; quantity/pack count
agreement and conflicts, plus comparable price relative gaps and currency checks.

**Explicit leakage protection:** `label`, `pair_id`, `id_left`,
`id_right`, `cluster_id_*`, `is_hard_negative`, and any test metadata are
**not features**. Hard-negative flag is used only for evaluation diagnostics.

## Models

- Logistic regression: train-only standardization; deterministic solver.
- LightGBM: fixed hyperparameters; `random_state=42`, one thread and
  deterministic histogram option. LightGBM may print version-specific warnings;
  hyperparameters and version are documented in the runtime environment.

Both are fit on the training split; threshold for maximum validation F1 is
selected **only** on `validation_safe`. Model selection is by validation AP.

## Metrics and error analysis

- Precision, recall, F1, confusion counts at threshold 0.5.
- F1-maximizing threshold selected on the same development validation.
- Average precision (AP), a non-interpolated precision-recall summary.
- Per-model private CSV of misclassified pairs with titles and probability.
- Hard-negative false-positive rate where the source flag exists.
- Top model features ranked by coefficients/importances; note LightGBM split
  importance is not directly comparable with logistic standardized coefficients.

Model artifacts and reports are intentionally ignored by Git.

## Limitations

No real-data metrics are baked into this repository or PR. Run locally against
the verified WDC archive; comparison numbers depend on your local dataset and
dependency versions. Validation-tuned F1 is optimistic because the same
validation rows select the threshold. Official test sets stay sealed for final
reporting. Product titles can contain noisy weights or multi-pack information;
the extractor is a starter baseline, not a complete product ontology.

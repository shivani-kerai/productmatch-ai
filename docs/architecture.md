# Architecture and implementation boundary

## Status

Session 1 implements only the local Python package, preprocessing, lexical features,
unit tests and CI. There is **no trained model, candidate index, model API, reviewer
queue, database, cloud deployment or monitoring system yet**.

## Target workflow

```text
Source catalog(s)
      |
Dataset ingestion -> schema checks -> normalizing product attributes
      |                                  |
      +----------- lexical retrieval ---+
      |                                  |
      +----------- embedding retrieval -+
                                         |
                                  merged candidates
                                         |
                              supervised reranker
                                         |
                            calibrated scores
                                         |
                       +-----------------+-----------------+
                       |                 |                 |
                   auto-match        human review       no match
                       |                 |                 |
                       +-----------------+-----------------+
                                         |
                                review event store
                                         |
                              monitoring / retraining
```

## Evaluation boundaries

- **Pairwise classification:** evaluate on official WDC pairwise train/validation/test splits.
- **Retrieval:** create an independently specified catalog/query benchmark; selected
  labeled pairs are not enough to establish catalog Recall@K.
- **Model decision:** choose accept/review/reject thresholds using validation only.
- **Production:** version scoring rules and preserve audit records for reviewer feedback.

## Planned components

1. Session 2: WDC schema-aware ingestion, dataset validation, EDA and lexical baseline.
2. Week 2: engineered features, logistic regression and LightGBM.
3. Week 3: lexical and dense candidate retrieval, FAISS, Recall@K.
4. Week 4: reranking, score calibration, selective prediction and review feedback.
5. Week 5: FastAPI, Streamlit, PostgreSQL (if needed), Docker, deployment and monitoring.

The basic CLI is **not** a product identity classifier; its output is inspection data.

# ProductMatch AI — System architecture

The following is an implementation architecture, not proof of cloud
hosting or end-to-end benchmark quality.

```mermaid
flowchart TD
    WDC[Official WDC pairwise source] --> Ingest[Validated ingestion and leakage audit]
    Ingest --> Train[Train 6000 pairs]
    Ingest --> Safe[Safe validation 3473 pairs]
    Train --> Model[Logistic Regression / LightGBM]
    Safe --> Threshold[Development model/threshold selection]
    Model --> Threshold
    Threshold --> Artifact[Frozen model artifact + SHA256]
    Safe --> Cal[Optional development-only sigmoid calibration]
    Artifact --> Cal
    Cal --> Bound[Model-bound calibration artifact]
    Catalog[Operator-provided catalog CSV] --> BM25[BM25 candidate retrieval]
    Titles[Query product title] --> BM25
    BM25 --> Candidates[Candidate offers]
    Candidates --> Serve[FastAPI search / pair score]
    Artifact --> Serve
    Bound --> Serve
    Serve --> Review[Streamlit reviewer UI]
    Review --> SQLite[SQLite review records]
    Serve --> Metrics[Authenticated route and latency metrics]
    External[Independent human-reviewed holdout] --> Audit[Offer-ID + provenance checks]
    Train --> Audit
    Safe --> Audit
    Audit --> Eval[Frozen holdout evaluation]
    Artifact --> Eval
    Eval --> Evidence[Evidence-gated portfolio report]
    Metrics --> Evidence
```

## Key boundaries

1. **Train:** fit predictive model only using the WDC training pairs.
2. **Development:** the safe validation set is used for model and threshold
   selection; its optional calibration is *not* independently validated.
3. **Sealed WDC tests:** neither official seen nor unseen test split is
   opened in the development/calibration/evidence pipeline.
4. **External evaluation:** independently collected labeled offers must
   pass ID-disjointness checks against both development splits. Source
   independence and label quality require manual audit.
5. **Serving:** local-only operator-selected model artifacts and catalog.
   Authentication is mandatory; human reviews do not retrain models.
6. **Operational evidence:** counts/latency only, avoiding product titles,
   identifiers, API keys and reviewer comments.

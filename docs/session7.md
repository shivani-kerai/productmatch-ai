# Session 7 — Evidence-based evaluation and release readiness

**Scope:** independent *user-supplied* pairwise holdout evaluation, repository
reproducibility, local service smoke checking and portfolio evidence reporting.
No WDC official test file is opened and no public service is deployed.

## A. Independent pairwise evaluation

Session 3 trained a frozen model on 6,000 WDC train pairs and selected
a threshold on a 3,473-pair leakage-safe validation subset. Session 6 has a
development-only calibration mapping because that subset lacks enough
independent offer-connected components for a disjoint calibration/audit split.

The Session 7 command evaluates frozen artifacts on **external labels**:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_independent.py \`
  --pairs-csv data/processed/external/holdout.csv \`
  --provenance-json data/processed/external/provenance.json
```

The required CSV fields are `id_left,id_right,title_left,title_right,label`.
Optional offer attributes include `brand_left,brand_right`,
`description_left,description_right`, `price_left,price_right` and
`priceCurrency_left,priceCurrency_right`.

The JSON provenance file should declare the collection and annotation protocol:

```json
{
  "label_origin": "human_reviewed",
  "source": "separately collected catalog or independent annotator dataset",
  "annotation_protocol": "explain positive same-sellable-item rules, size, variant and pack policies",
  "labeler_count": 2,
  "independence_statement": "describe why no development examples or model suggestions were reused"
}
```

These fields are **operator assertions**, not external verification. Keep the
data and full manifest local. The evaluation rejects duplicate/reversed pairs,
self pairs, missing/one-class labels, and any offer-ID overlap with
`train.parquet` or `validation_safe.parquet`.

With a trusted Session 6 calibration mapping:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_independent.py \`
  --pairs-csv data/processed/external/holdout.csv \`
  --provenance-json data/processed/external/provenance.json \`
  --calibration-path artifacts/session6_calibration/sigmoid.joblib
```

Output: `reports/session7/independent_pairwise.json`. Metrics are frozen-
threshold precision, recall, F1, non-interpolated average precision, Brier
score, fixed-bin ECE and ROC-AUC; no score calibration or optimization
is performed on the external evaluation examples.

**Note:** good ID-disjointness is necessary but not sufficient. It does not
verify labeling quality, statistical independence, or repeated real-world
products with different IDs. Human audit is still required.

Synthetic demo labels are rejected by default. Explicit `--allow-demo`
permits scripted fixtures only and always marks metrics synthetic.

## B. Retrieval evidence

Session 4's default retrieval `known_positive_recovery_at_k` uses positive
pair labels and has **incomplete** catalog relevance judgments.
It must not be reported as Recall@K. For genuine catalog Recall@K, curate an
independent and completely assessed query/catalog relevance table, and
document the labeling policy and cardinality.

## C. Local service smoke test

Start the Session 6 service bound to `127.0.0.1`, using local operator-owned
artifacts and two random tokens. In a separate PowerShell session, configure
the same API key (never print it or commit it) and run:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_local_api.py
```

This utility accepts **only** plain-HTTP localhost or 127.0.0.1 origins with
explicit ports. It never follows redirects and performs read-only probes
of liveness, authenticated readiness, candidate search (if catalog ready),
model scoring (if model ready), metrics and reviewer-token enforcement.
It writes `reports/session7/local_smoke.json` without secrets, request
payloads, reviewer notes or product titles.

A smoke success is **not** load testing, cloud deployment or security sign-off.

## D. Portfolio reporting without invented results

Generate honest placeholders:

```powershell
.\.venv\Scripts\python.exe scripts/build_portfolio_report.py
```

After generating reports, include them:

```powershell
.\.venv\Scripts\python.exe scripts/build_portfolio_report.py \`
  --pairwise reports/session7/independent_pairwise.json \`
  --retrieval reports/session4_retrieval/retrieval_metrics.json \`
  --smoke reports/session7/local_smoke.json
```

Outputs remain gitignored:
- `reports/session7/evidence.json`
- `reports/session7/portfolio_summary.md`

A synthetic fixture is explicitly labeled as such. Human-reviewed label
provenance is operator-asserted and not verified by the software. An
incomplete retrieval judgment set stays labeled known-positive recovery.
A local smoke test never becomes a hosted URL or production-readiness claim.

## E. Engineering boundaries and future work

- No official WDC seen/unseen test metrics are generated in Session 7.
- No external dataset is silently downloaded or committed.
- No calibration or decision threshold is refit on independent evaluation.
- SHA-256 records exact model, calibration and external evidence bytes.
- Add confidence intervals, subgroup analysis, drift and load testing after
  real independent annotations and a complete catalog benchmark are available.
- Deploy only after an explicit infrastructure choice, approval, security
  review, managed secrets, private networking and appropriate monitoring.

# Session 6 — Calibration diagnostics, model integrity and monitoring

## Goals and scope

This milestone hardens **local prototype operation** without claiming an
unbiased production calibration, a live cloud deployment, or final official WDC
test results. No official WDC seen/unseen test parquet file is opened.

## Development-only calibration

The optional `scripts/calibrate_model.py` command loads:
- `data/processed/wdc50_medium/validation_safe.parquet`,
- its provenance `manifest.json`, and
- the trusted frozen Session 3 model artifact.

It computes frozen-model raw probabilities, forms offer-connected components
from pair IDs, and uses `GroupShuffleSplit` to create disjoint
calibration/audit partitions. A deterministic search refuses to proceed if
there are too few independent components or either partition lacks both
classes. No pairwise label, offer ID, pair ID or test metadata becomes a
predictive feature.

A logistic sigmoid is fitted on raw-score log-odds in the calibration partition;
a threshold is chosen on the **same calibration partition**. The audit partition
reports Brier score, expected calibration error (10 fixed bins), classification
metrics and the applied threshold.

**Limitation:** Session 3 already used the entire `validation_safe` set to
select and tune its original classifier. Thus these calibration/audit
partitions are secondary development diagnostics, NOT a clean externally
independent evaluation. A fresh labeled dataset is needed for production
claims. No fabricated experimental results are committed.

Run in Windows PowerShell:

```powershell
cd C:\productmatch-ai
git fetch origin
git switch feature/session-6-calibration-monitoring
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data,models,service,ui]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/calibrate_model.py
```

Generated artifacts are gitignored:
- `artifacts/session6_calibration/sigmoid.joblib`
- `reports/session6_calibration/reliability.json`

The artifact records SHA-256 of the **exact model file**, the frozen feature
order, sigmoid coefficients and a calibration-selected threshold.
The API checks the model checksum and fails startup on mismatch.
Only load model and calibration joblib files from trusted local operators;
pickle/joblib is unsafe for untrusted files.

## Local API setup

Configure independent, strong secrets in PowerShell or an ignored .env file.
The API requires both tokens, even if the reviewer UI is unused. A locally
generated key can be made with:

```powershell
$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$bytes = New-Object byte[] 32
$rng.GetBytes($bytes)
$env:PRODUCTMATCH_API_KEY = [BitConverter]::ToString($bytes).Replace("-", "")
$rng.GetBytes($bytes)
$env:PRODUCTMATCH_REVIEW_TOKEN = [BitConverter]::ToString($bytes).Replace("-", "")
$rng.Dispose()
$env:PRODUCTMATCH_CATALOG_CSV = "examples/catalog_demo.csv"
$env:PRODUCTMATCH_MODEL_PATH = "artifacts/session3_baselines/logistic_regression.joblib"
$env:PRODUCTMATCH_CALIBRATION_PATH = "artifacts/session6_calibration/sigmoid.joblib"
$env:PRODUCTMATCH_REVIEW_DB = "runtime/reviews.sqlite3"
.\.venv\Scripts\python.exe -m uvicorn productmatch.api.service:app --host 127.0.0.1 --port 8000
```

Use the **same** model path as when calibration was fitted. Any byte change
means calibration must be regenerated.

## Monitoring

`GET /metrics` requires `X-API-Key` and returns Prometheus-compatible
request counters grouped by known route/method/status class, plus cumulative
request duration histograms. Unknown paths all become `other` to avoid
unbounded label cardinality and leaking user-provided URL text.

Do NOT emit raw payloads, model feature values, titles, reviewer notes, API
keys or token values in metrics. Monitoring is per Python process; in
multi-worker environments an external aggregation system is necessary.
Prometheus-style metrics here are a local diagnostic, not an observability
infrastructure deployment.

`GET /ready` additionally provides a 12-character model SHA-256 fingerprint,
a boolean calibrated status, catalog and model readiness. It requires the API
key. Readiness is not equivalent to a deployment SLA or complete end-to-end
latency/load testing.

## Container safeguards

Compose binds HTTP to `127.0.0.1:8000` only. Security options include
non-root execution (in Dockerfile), read-only root filesystem, dropped Linux
capabilities, no-new-privileges, PID limit and a bounded temporary filesystem.

This does not make the application Internet-safe. SQLite needs a writable host
volume owned by the container UID; prepare `runtime` with correct permissions.
Model and corpus mounts remain read-only. Do not put tokens in version control,
expose `/metrics` or `/docs` publicly, or deploy without a security review.

## Blockers before real production

- Independent score calibration/holdout data and complete retrieval judgments.
- Managed identity, TLS and authorization roles instead of static API keys.
- Rate limiting, reverse proxy, request bounds and attack-surface review.
- Signed, immutable and audited model artifacts.
- Multi-worker telemetry aggregation, tracing, retention and alerting.
- Database migration strategy, backups and reviewer access governance.
- Load tests and deployment approvals.

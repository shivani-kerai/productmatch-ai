# Session 5 — API service, reviewer UI and local deployment

## Components

- `src/productmatch/api/service.py`: FastAPI service factory, authentication,
  readiness, candidate search, pairwise matching, and authenticated review routes.
- `src/productmatch/api/runtime.py`: operator-selected catalog CSV and **trusted**
  local Session 3 model, validated feature ordering, batch probability scoring.
- `src/productmatch/api/schemas.py`: strict request contracts and size limits.
- `src/productmatch/api/store.py`: SQLite feedback persistence, no auto-retraining.
- `app/reviewer.py`: Streamlit local reviewer and model inspection workflow.
- `examples/catalog_demo.csv`: fabricated catalog for an offline BM25 demo.
- `Dockerfile`, `compose.yaml`, `.env.example`: local container example.

## Local Windows usage

In PowerShell, switch to the Session 5 branch and install dependencies:

```powershell
git fetch origin
git switch feature/session-5-api-review-ui
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data,models,service,ui]"
```

Generate independent secrets in the PowerShell terminal that will run the API:

```powershell
$env:PRODUCTMATCH_API_KEY = [Convert]::ToHexString([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
$env:PRODUCTMATCH_REVIEW_TOKEN = [Convert]::ToHexString([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
$env:PRODUCTMATCH_CATALOG_CSV = "examples/catalog_demo.csv"
$env:PRODUCTMATCH_REVIEW_DB = "runtime/reviews.sqlite3"
# Optional, only after Session 3 training with trusted local data:
# $env:PRODUCTMATCH_MODEL_PATH = "artifacts/session3_baselines/logistic_regression.joblib"

.\.venv\Scripts\python.exe -m uvicorn productmatch.api.service:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for Swagger UI (requests still need API keys).

Start Streamlit in a *second* terminal:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/reviewer.py --server.address 127.0.0.1
```

Paste both secrets from the API terminal into the Streamlit sidebar.
The HTTP server only listens on loopback.

## Endpoints

| HTTP method | Endpoint | Authentication | Meaning |
|---|---|---|---|
| GET | `/health` | none | Process liveness, not model readiness |
| GET | `/ready` | X-API-Key | Model/catalog availability |
| POST | `/v1/search` | X-API-Key | BM25 candidates or optional supervised rerank |
| POST | `/v1/match` | X-API-Key | Local Session 3 model probability |
| POST | `/v1/reviews` | X-API-Key + X-Review-Token | Save human review |
| GET | `/v1/reviews` | X-API-Key + X-Review-Token | List recent review records |

Missing model/catalog returns 503 for dependent endpoints. Service startup
requires both secrets. External users do not select local artifact or data paths.

## Local-only Docker example

Copy `.env.example` to `.env` and replace all token placeholders with
strong random secrets; ensure the host has `artifacts/` and `data/processed/`
prepared. In the repo root run `docker compose up --build`.

The Compose example binds the host port to **127.0.0.1**, not public interfaces.
The container's internal port is 8000 and the non-root process writes to a
local runtime directory. The default paths in the example assume you've
prepared the Session 3 trusted model and a catalog CSV on the host.

Never commit the populated `.env`, model joblib, SQLite or source datasets.
Do not treat this prototype container as hardened production infrastructure.

## Known limitations

- A Session 3 model threshold was optimized on development validation; it is
  not calibrated for real business acceptance decisions.
- Search does a deterministic BM25 scan of a small local catalog. This example
  is not a scalable vector database or distributed retrieval service.
- The human feedback table is for manual review, not an autonomous feedback
  retraining loop. SQLite lacks multi-tenant roles/migrations.
- API-key auth is for loopback development; prefer managed identity/roles,
  network access control and HTTPS in a real environment.
- The Streamlit UI is for trusted local users; do not expose it publicly.
- Model loading uses pickle-based joblib: only operator-owned, trusted files.
- No live hosted URL is claimed by this branch or pull request.

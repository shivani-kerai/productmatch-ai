# ProductMatch AI

**A portfolio project for a hybrid Data Scientist + ML Engineer role.**

This project will grow into an end-to-end system for identifying when two e-commerce listings
represent the **same sellable product**. It will combine product attribute normalization,
lexical and semantic retrieval, supervised pairwise matching, confidence calibration,
human review, API serving, evaluation, and monitoring.

> **Current status: Session 1 starter (v0.1.0).** The repository is runnable and tested,
> but **does not contain a trained matching model, a production match decision, or WDC
> benchmark data**. The demonstration CLI computes simple text features only.

## What works right now

- Installable `src/` Python package and `productmatch` command-line entry point.
- Unicode-aware title normalization (including `250ML` -> `250 ml`).
- Token Jaccard similarity and an exact-normalized-title feature.
- Automated unit tests, linting, formatting checks, and GitHub Actions CI.
- Design documents, a data card, and a transparent project roadmap.

## Prerequisites

- Windows 10/11 with **Python 3.11**, Git, and PowerShell.
- Recommended: VS Code, GitHub CLI (`gh`) for easy publication.
- No API keys, GPU, cloud account, or WDC dataset needed for Session 1.

## Quick start — Windows PowerShell (no activation required)

Clone the published repository, open PowerShell, and change into the folder
containing `pyproject.toml`:

```powershell
git clone https://github.com/shivani-kerai/productmatch-ai.git
cd productmatch-ai

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Verify the CLI, tests, formatting and linting
.\.venv\Scripts\python.exe -m productmatch --help
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

**Note:** You do not have to run `Activate.ps1`; invoking the virtual environment's
Python executable directly also avoids PowerShell execution-policy issues.

Alternatively, while in the repository root:

```powershell
.\scripts\setup_windows.ps1
.\scripts\check_windows.ps1
```

If local PowerShell policy blocks `.ps1` scripts, use the direct commands above.

### Try the sample commands

```powershell
.\.venv\Scripts\python.exe -m productmatch normalize "Dove Deep Moisture Body-Wash 250ML"
.\.venv\Scripts\python.exe -m productmatch features "Dove Body Wash 250ml" "Dove Body Wash 500 ml"
```

The first command returns normalized text. The second returns simple lexical features
in JSON. **A high similarity is NOT a reliable product-match prediction:** size,
pack count, model number, and variant checks will be incorporated later.

### Optional: activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
productmatch --help
```

## Installing future data-science dependencies

When you begin the WDC benchmark ingestion and baseline model (Session 2 onward):

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data]"
```

The initial installation intentionally stays lightweight. Never commit downloaded
benchmark data, local databases, credentials, or large model artifacts.

## Contribute to the GitHub repository

The repository is available at https://github.com/shivani-kerai/productmatch-ai.
After making a local change, commit and push it to a feature branch:

```powershell
git switch -c feature/session-2-wdc-ingestion
git add .
git commit -m "feat: add WDC ingestion pipeline"
git push -u origin feature/session-2-wdc-ingestion
```

If `git commit` requests an identity, set your own Git username and email with
`git config --global user.name "Your Name"` and
`git config --global user.email "you@example.com"` (replace with your actual details).

## Proposed five-week milestones

| Week | Focus | Deliverable |
| --- | --- | --- |
| 1 | Project setup, WDC data ingestion, lexical baseline | Validated datasets, baseline report, tests |
| 2 | Pairwise features + supervised classifiers | Logistic regression / LightGBM comparison |
| 3 | Candidate generation | BM25/embedding retrieval and Recall@K evaluation |
| 4 | Reranking + calibration + review workflow | Evaluated, selective matching policy |
| 5 | FastAPI, Streamlit, Docker, deployment, CI/CD | Deployed demo and model card |

The WDC **pairwise** benchmark is appropriate for classification evaluation.
Catalog retrieval needs its own separate labeled query/catalog evaluation protocol;
a selected set of positive and negative pairs is not automatically a retrieval benchmark.

## Dataset (Session 2)

Official benchmark: https://webdatacommons.org/largescaleproductcorpus/wdc-products/index.html

Prefer the WDC Products pairwise benchmark, with a specified corner-case ratio and
its official train/validation/seen/unseen splits. Verify the archive names, field
schema, provenance and permitted use before writing the ingestion script.

See [data card](docs/data_card.md), [architecture](docs/architecture.md),
[model card](docs/model_card.md), and [next steps](docs/next_steps.md).

## Quality checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=productmatch --cov-report=term-missing
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

CI runs these checks for Python 3.11, 3.12, and 3.13 on pushes and pull requests.
It does not train a model, download benchmark data, or deploy anything.

## Data safety and reproducibility

- Never use leaked IDs, cluster labels, or ground-truth labels as predictive features.
- Respect the official benchmark splits: do not randomly reshuffle pairs into test data.
- Select thresholds on validation data only; reserve official test splits for reporting.
- Record dataset revision, experiment configuration, environment and model version.
- Do not upload proprietary employer data or credentials.

## License

Project code is MIT-licensed (see `LICENSE`). **Third-party datasets, pretrained models,
and external content retain their own usage terms and are not covered by this license.**

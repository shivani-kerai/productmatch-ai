# Session 2 — Benchmark ingestion, validation, EDA

**Branch:** `feature/session-2-wdc-ingestion` (PR for review)

## Scope delivered

- `scripts/download_wdc.py`: bounded, atomic download from the official 50pair.zip URL; archives are never committed.
- `src/productmatch/data/ingestion.py`: enumerate ZIP or extracted file members, select medium pairwise splits, stream compressed JSON Lines.
- `src/productmatch/data/validation.py`: strict row-level checks, IDs, unique unordered pairs, binary labels, optional attributes, test-specific leakage logic.
- `src/productmatch/data/pipeline.py`: preserve official splits and row order, normalize titles, write four Parquet files and an input/provenance manifest.
- `src/productmatch/data/eda.py`, `scripts/run_eda.py` and `notebooks/01_wdc_eda.ipynb`: class balance, missingness, title lengths, negative hardness.
- `tests/test_wdc_validation.py` and `tests/test_wdc_pipeline.py`: offline synthetic fixture tests.

## Run on Windows (PowerShell)

```powershell
git switch feature/session-2-wdc-ingestion
.\.venv\Scripts\python.exe -m pip install -e ".[dev,data]"
.\.venv\Scripts\python.exe scripts/download_wdc.py
.\.venv\Scripts\python.exe scripts/prepare_data.py --list-members
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/run_eda.py
.\.venv\Scripts\python.exe -m pytest -q
```

Official data (50% corner cases, medium pairwise) has 6,000 training, 3,500 validation and 4,500 pairs per test variant. The pipeline refuses unexpected row counts by default. For *synthetic local fixtures only*, pass `--skip-official-count-check`.

## Resulting local files (gitignored)

```text
data/raw/wdc/50pair.zip
data/processed/wdc50_medium/
    train.parquet
    validation.parquet
    test_seen.parquet
    test_unseen.parquet
    manifest.json
reports/wdc_eda/
    summary.json
    label_balance.png
    train_missingness.png
```

### Why no random train/test split?

The official benchmark separates offer records between training, validation and test. Randomly splitting labeled pairs would reintroduce repeated product offers, inflate apparent performance, and break comparability with published benchmark results.

### Why not compare the two test variants against each other for leakage?

The same held-out offer can appear in alternative seen/unseen **test configurations**. We check train–validation, train–each test and validation–each test, while allowing reuse across alternative test configurations.

### Limitations and follow-up

The official archive could not be fetched in the execution environment used to author this PR. The implementation is verified against offline, WDC-shaped synthetic JSONL gzip fixtures, but **the exact official archive member names are not confirmed here**. The `--list-members` mode and explicit member overrides provide a safe fallback. Do not claim real benchmark metrics until running on the real data.

**Dataset:** https://webdatacommons.org/largescaleproductcorpus/wdc-products/index.html

**Upstream creation code:** https://github.com/wbsg-uni-mannheim/wdcproducts


## Diagnosing cross-split offer overlap

If `prepare_data.py` raises `DatasetValidationError` for shared IDs between
validation and a test split, **do not disable leakage checks or randomly split data**.
The WDC Products documentation states that offer records should be disjoint across
training, validation and test.

After `git pull` on the feature branch, run:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_wdc_overlap.py
```

The script lists the four selected archive members, their row counts and all
cross-split shared IDs. For a limited display of titles of overlapping offers,
add `--show-titles`. It does not write files or alter any data.

Check that the paths come from the same WDC Products 50% pairwise configuration:
`wdcproducts50cc50rnd000un_train_medium`,
`wdcproducts50cc50rnd000un_valid_medium`, and the
`000un` and `100un` gold standards.

Do not proceed to a leakage-removal workaround until the exact paths and IDs have
been inspected. Filtering validation pairs would change the official benchmark
and must be recorded explicitly if later chosen.


## Known official-data overlap and safe development protocol

The WDC 50% medium pairwise archive has been observed to include **four shared
validation vs. 000un seen-test offer IDs** despite the published disjointness claim:
`19015384`, `28830600`, `36532408`, `51902182`.

Strict preparation intentionally refuses to continue with this overlap. Once you
have inspected the member list and verified the source intersections with
`scripts/inspect_wdc_overlap.py`, explicitly run:

```powershell
.\\.venv\\Scripts\\python.exe scripts/prepare_data.py --quarantine-validation-overlap
.\\.venv\\Scripts\\python.exe scripts/run_eda.py
```

The pipeline keeps the **original** `validation.parquet` (3,500 rows), both
unmodified test splits and training data. It also writes `validation_safe.parquet`
containing only validation pairs with **neither offer ID shared with either test
variant**. No test labels, predictions, scores or product text are used to select
quarantined validation rows. Only offer IDs are compared. The number of dropped
validation pairs and exact IDs are saved in `manifest.json`. The EDA script
*requires* and uses this safe validation copy.

For fair reporting, future hyperparameter selection and threshold calibration must
use `validation_safe.parquet`, never contaminated `validation.parquet`. Report
that the experimental validation subset differs from the official WDC validation
set; do not claim directly comparable official validation scores.

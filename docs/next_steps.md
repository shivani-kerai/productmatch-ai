# Session 2: WDC ingestion and EDA — implementation plan

Once Session 1 quality checks pass:

1. Use the **official** WDC Products benchmark page to select and download one
   **pairwise** benchmark configuration. Do not silently swap in the older WDC LSPM
   dataset; it is a different resource with potentially different schemas.
2. Record archive filename, download URL, checksum, extraction method and license.
3. Inspect actual raw columns, nested product objects, label values and data types.
4. Implement `src/productmatch/data/ingestion.py` and `validation.py` based on the
   verified real schema. Do not hardcode guessed raw column names.
5. Implement `scripts/prepare_data.py`, generating local processed Parquet files.
6. Create `notebooks/01_data_exploration.ipynb` for label distribution, attributes,
   missingness, brand, pack sizes and hard negatives.
7. Build a lexical pairwise baseline and log validation metrics.
8. Add deterministic unit tests with tiny fabricated fixtures (not copyrighted data).
9. Compare default-threshold and tuned-threshold precision/recall and F1; tune only
   on validation. Keep official test splits sealed until final reporting.
10. Submit a small pull request with a summary of data-quality findings.

**Definition of done:** a documented dataset manifest, reproducible preparation
script, data validation tests, and a baseline evaluation report.

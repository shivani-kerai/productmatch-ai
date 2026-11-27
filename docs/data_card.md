# Dataset card (planned)

## Intended benchmark

- Name: WDC Products — multi-dimensional entity matching benchmark.
- Official source: https://webdatacommons.org/largescaleproductcorpus/wdc-products/index.html
- Planned task: pairwise classification using an official dataset variant.
- Intended starting setup: 50% corner cases, medium training set (subject to confirming
  official archive names and column schema when downloaded).
- Data status: **NOT DOWNLOADED** in Session 1; no samples or ground truth included.

## Session 2 implementation

- Canonical schema: `pair_id`, `label`, `id_left`, `id_right`, left/right titles, brands, descriptions, prices, currencies; optional `cluster_id` and `is_hard_negative` for audit only.
- Loader reads `.json.gz` members within the official `50pair.zip` without extracting the ZIP.
- Official medium pairwise split sizes are checked by default: 6,000 train / 3,500 validation / 4,500 per test variant.
- Output: Parquet splits and a deterministic provenance manifest. All generated artifacts are gitignored.
- Full archive has **not been downloaded or executed in this hosted environment**; schema-level behavior is covered by synthetic fixtures.

## Data governance checklist

- [ ] Confirm license, attribution obligations and redistribution rights.
- [ ] Record download URL, date, file sizes, and SHA256 checksums.
- [ ] Inspect actual field names and types (do not assume schema).
- [ ] Record original label semantics and class prevalence.
- [ ] Preserve official splits; do not re-split rows at random.
- [ ] Test for duplicate pairs and overlap/leakage across split identities.
- [ ] Document missingness, brand imbalance and hard-negative cases.
- [ ] Do not publish source data until rights are confirmed.

## Known caveats

Pairwise classification and catalog retrieval require different evaluation designs.
Potential benchmark annotation errors should be surfaced as uncertainty, not silently
re-labeled using model predictions.

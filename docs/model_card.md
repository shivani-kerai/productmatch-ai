# Model card — Session 3 development baselines

Two **pairwise** classifiers are implemented: standardized logistic regression
and LightGBM trained on supervised WDC Products medium pairwise training data.
No benchmark validation scores are published in this repository; run the
local experiment to produce evidence before citing performance in a portfolio.

**Data:** WDC Products, 50% corner-case pairwise benchmark.
Training: 6,000 pairs; validation: 3,473 pairs after an explicit 27-pair
validation quarantine caused by four overlapping official seen-test offer IDs.
The full original validation and both test variants remain unchanged.

**Input features:** 23 deterministic, numeric attributes from offer titles,
brand, descriptions, mass/volume quantities, pack counts, price and currency.
Never train on offer/pair/cluster IDs, ground truth, or hard-negative labels.

**Evaluation:** validation precision/recall/F1, average precision and
hard-negative false-positive rate; fixed and validation-tuned thresholds.
Official test labels must remain sealed until model selection is finalized.
Validation-optimized results cannot be represented as unbiased test performance.

**Privacy:** no proprietary product listings or raw benchmark data are
committed. Model artifacts and review CSVs are local and gitignored.

**Risks:** brand aliases, incomplete descriptions, model/version identifiers,
unit inconsistencies, variable-price sales and multilingual product titles can
degrade generalization. Treat errors on variants and pack sizes as particularly
high-risk false matches.

**Deployment:** no matching API, reviewer interface or production readiness
claim in Session 3.

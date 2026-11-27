"""Score a frozen ProductMatch model on an independently labeled CSV.

Requires a user-supplied provenance JSON. Demo labels are permitted only with
--allow-demo, and are always flagged synthetic in the generated report.
Never reads official WDC test sets or optimizes decisions on evaluation rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from productmatch.evaluation.external import evaluate_external_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    parser.add_argument(
        "--development-dir", type=Path, default=Path("data/processed/wdc50_medium")
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/session3_baselines/logistic_regression.joblib"),
    )
    parser.add_argument("--calibration-path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/session7/independent_pairwise.json"),
    )
    parser.add_argument(
        "--allow-demo", action="store_true", help="Allow explicitly synthetic fixture labels"
    )
    args = parser.parse_args()
    result = evaluate_external_pairs(
        pairs_csv=args.pairs_csv,
        provenance_json=args.provenance_json,
        development_dir=args.development_dir,
        model_path=args.model_path,
        calibration_path=args.calibration_path,
        report_path=args.output,
        allow_demo=args.allow_demo,
    )
    print(
        json.dumps(
            {
                "evaluation_type": result["evaluation_type"],
                "verification_status": result["verification_status"],
                "external_pairs": result["audit"]["external_pairs"],
                "train_offer_overlap": result["audit"]["train_offer_overlap"],
                "validation_offer_overlap": result["audit"]["validation_offer_overlap"],
                "metrics": result["metrics"],
                "report": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

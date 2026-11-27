"""Run deterministic supervised product matching baseline experiments.

Never reads official seen/unseen test sets. Requires Session 2 safe validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from productmatch.models.baselines import DEFAULT_SEED, MODEL_CHOICES
from productmatch.models.training import train_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/wdc50_medium")
    )
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports/session3_baselines")
    )
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/session3_baselines")
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    results = train_baselines(
        args.data_dir, args.report_dir, args.artifact_dir,
        model_names=args.models, seed=args.seed
    )
    print(
        json.dumps(
            {
                "train_pairs": results["train_pairs"],
                "validation_safe_pairs": results["validation_pairs"],
                "validation_excluded_pairs": results["source_validation_excluded_pairs"],
                "best_validation_model_by_ap": results["best_validation_model_by_ap"],
                "models": {
                    name: {
                        "average_precision": details["default_threshold_metrics"][
                            "average_precision"
                        ],
                        "f1_at_0_5": details["default_threshold_metrics"]["f1"],
                        "f1_at_tuned_threshold": details[
                            "validation_tuned_threshold_metrics"
                        ]["f1"],
                        "selected_threshold": details[
                            "validation_tuned_threshold_metrics"
                        ]["threshold"],
                    }
                    for name, details in results["models"].items()
                },
                "full_report": str(args.report_dir / "metrics.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

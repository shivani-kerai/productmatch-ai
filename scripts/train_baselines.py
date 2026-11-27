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

    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    results = train_baselines(

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

"""Create deterministic WDC EDA summaries and plots from train/validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from productmatch.data.eda import development_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/wdc50_medium")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/wdc_eda"))
    args = parser.parse_args()

    # Do not load test_seen.parquet or test_unseen.parquet during development.
    # Model-development EDA defaults to the quarantined validation copy
    # when present; do not silently inspect the contaminated official split.
    safe_validation = args.data / "validation_safe.parquet"
    if not safe_validation.is_file():
        raise FileNotFoundError(
            "validation_safe.parquet is required. Re-run scripts/prepare_data.py "
            "(add --quarantine-validation-overlap if source validation overlaps tests)."
        )
    frames = {
        "train": pd.read_parquet(args.data / "train.parquet"),
        "validation": pd.read_parquet(safe_validation),
    }
    summary = development_summary(frames)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, frame) in zip(axes, frames.items()):
        frame["label"].value_counts().reindex([0, 1], fill_value=0).plot.bar(ax=ax)
        ax.set(title=f"{name} label distribution", xlabel="0=nonmatch, 1=match",
               ylabel="Pairs")
    fig.tight_layout()
    fig.savefig(args.output / "label_balance.png", dpi=140)
    plt.close(fig)

    fields = summary["train"]["attribute_missing_fraction"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(list(fields), list(fields.values()))
    ax.set(xlabel="Missing fraction", xlim=(0, 1), title="Training attribute missingness")
    fig.tight_layout()
    fig.savefig(args.output / "train_missingness.png", dpi=140)
    plt.close(fig)
    print(f"Wrote development-only EDA to {args.output}")


if __name__ == "__main__":
    main()

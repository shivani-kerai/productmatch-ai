"""Discover, validate and prepare official WDC Products benchmark splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from productmatch.data.ingestion import available_members
from productmatch.data.pipeline import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/raw/wdc/50pair.zip"),
        help="Downloaded 50pair.zip or extracted root directory",
    )
    parser.add_argument("--output", type=Path, default=Path("data/processed/wdc50_medium"))
    parser.add_argument(
        "--list-members",
        action="store_true",
        help="Print archive JSONL names to identify the exact official filenames",
    )
    parser.add_argument("--train-member")
    parser.add_argument("--validation-member")
    parser.add_argument("--test-seen-member")
    parser.add_argument("--test-unseen-member")
    parser.add_argument(
        "--quarantine-validation-overlap",
        action="store_true",
        help=(
            "Keep all official splits unchanged and additionally write a "
            "validation_safe.parquet with validation pairs involving "
            "test-overlapping offer IDs excluded; record the exclusions."
        ),
    )
    parser.add_argument(
        "--skip-official-count-check",
        action="store_true",
        help="Allow small development fixtures, NOT recommended for actual WDC",
    )
    args = parser.parse_args()
    if args.list_members:
        print("\n".join(available_members(args.source)))
        return

    overrides = {
        role: value
        for role, value in {
            "train": args.train_member,
            "validation": args.validation_member,
            "test_seen": args.test_seen_member,
            "test_unseen": args.test_unseen_member,
        }.items()
        if value is not None
    }
    manifest = prepare_dataset(
        args.source,
        args.output,
        member_overrides=overrides,
        enforce_official_counts=not args.skip_official_count_check,
        quarantine_validation_overlap=args.quarantine_validation_overlap,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

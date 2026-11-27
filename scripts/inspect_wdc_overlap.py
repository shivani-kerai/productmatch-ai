"""Inspect WDC pair-wise archive member selection and offer-level overlap.

Read-only diagnostic. This command never writes Parquet, changes splits, or
trains models. Print only IDs by default; add --show-titles when necessary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from productmatch.data.ingestion import (
    available_members,
    read_jsonl_member,
    select_split_members,
)
from productmatch.data.validation import find_offer_overlaps, validate_pairs


def _related_pairs(frame: pd.DataFrame, offer_id: int) -> pd.DataFrame:
    """Return rows containing the given offer ID on either side."""
    return frame.loc[(frame["id_left"] == offer_id) | (frame["id_right"] == offer_id)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/wdc/50pair.zip"))
    parser.add_argument("--train-member")
    parser.add_argument("--validation-member")
    parser.add_argument("--test-seen-member")
    parser.add_argument("--test-unseen-member")
    parser.add_argument("--show-titles", action="store_true")
    args = parser.parse_args()

    overrides = {
        key: value
        for key, value in {
            "train": args.train_member,
            "validation": args.validation_member,
            "test_seen": args.test_seen_member,
            "test_unseen": args.test_unseen_member,
        }.items()
        if value is not None
    }
    selected = select_split_members(available_members(args.source), overrides)
    print("Selected WDC archive members (read-only):")
    for role, member in selected.items():
        print(f"  {role}: {member}")

    splits = {
        role: validate_pairs(read_jsonl_member(args.source, member), role)
        for role, member in selected.items()
    }
    print("\nLoaded pair counts:")
    for role, frame in splits.items():
        print(f"  {role}: {len(frame)} pairs")

    report = find_offer_overlaps(splits)
    print("\nOffer-ID intersections:")
    for name, ids in report.items():
        print(f"  {name}: {len(ids)} shared offer IDs")
        if not ids:
            continue
        print(f"    IDs (up to 20): {ids[:20]}")
        left, right = name.split("_vs_")
        for offer_id in ids[:5]:
            for role in (left, right):
                related = _related_pairs(splits[role], offer_id)
                pair_ids = related["pair_id"].astype(str).head(5).tolist()
                print(f"    {offer_id} in {role}: pair_ids={pair_ids}")
                if args.show_titles:
                    for _, row in related.head(2).iterrows():
                        side = "left" if int(row["id_left"]) == offer_id else "right"
                        print(f"      title={str(row[f'title_{side}'])[:120]}")
    if any(report.values()):
        print(
            "\nWARNING: Cross-split offer overlap found. No rows were changed. "
            "Inspect selected files before preparing data."
        )
    else:
        print("\nNo development/test offer overlap found.")


if __name__ == "__main__":
    main()

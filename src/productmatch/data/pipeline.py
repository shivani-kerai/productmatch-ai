"""Deterministic processing of official WDC Products pairwise splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from productmatch.data.ingestion import (
    available_members,
    file_sha256,
    read_jsonl_member,
    select_split_members,
)
from productmatch.data.preprocessing import normalize_title
from productmatch.data.validation import (
    DatasetValidationError,
    audit_offers_across_splits,
    validate_pairs,
)

EXPECTED_COUNTS = {
    "train": 6000,
    "validation": 3500,
    "test_seen": 4500,
    "test_unseen": 4500,
}
PIPELINE_VERSION = "session2-v1"


def prepare_dataset(
    source: Path,
    output_dir: Path,
    *,
    member_overrides: dict[str, str] | None = None,
    enforce_official_counts: bool = True,
) -> dict[str, object]:
    """Prepare existing benchmark splits; NEVER create random row partitions.

    Validate all four splits and their offer-level boundaries before writing
    any output files. Do not tune models or select thresholds on test splits.
    """
    source = Path(source)
    output_dir = Path(output_dir)
    members = select_split_members(available_members(source), member_overrides)
    splits: dict[str, pd.DataFrame] = {}

    for role, member in members.items():
        df = validate_pairs(read_jsonl_member(source, member), role)
        if enforce_official_counts and len(df) != EXPECTED_COUNTS[role]:
            raise DatasetValidationError(
                f"{role}: expected {EXPECTED_COUNTS[role]} pairs, got {len(df)} "
                "(confirm this is the official 50% corner-case medium pairwise setup)"
            )
        for side in ("left", "right"):
            df[f"title_{side}_normalized"] = df[f"title_{side}"].map(normalize_title)
        splits[role] = df

    overlaps = audit_offers_across_splits(splits)
    source_digest = (
        file_sha256(source) if source.is_file() else
        {role: file_sha256(source / member) for role, member in members.items()}
    )
    manifest: dict[str, object] = {
        "pipeline_version": PIPELINE_VERSION,
        "benchmark": "WDC Products (50% corner cases, pairwise medium development)",
        "source": "https://webdatacommons.org/largescaleproductcorpus/wdc-products/",
        "archive_sha256": source_digest,
        "member_paths": members,
        "splits": {
            role: {
                "rows": len(df),
                "positive": int(df["label"].sum()),
                "negative": int((df["label"] == 0).sum()),
                "unique_offers": len(
                    set(df["id_left"].tolist()) | set(df["id_right"].tolist())
                ),
            }
            for role, df in splits.items()
        },
        "offer_overlap_audit": overlaps,
        "data_policy": "official splits preserved; no test-based tuning",
        "expected_official_counts_checked": enforce_official_counts,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for role, df in splits.items():
        df.to_parquet(output_dir / f"{role}.parquet", index=False, engine="pyarrow")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

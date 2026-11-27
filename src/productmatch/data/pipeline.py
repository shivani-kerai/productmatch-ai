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
    find_offer_overlaps,
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
    quarantine_validation_overlap: bool = False,
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

    # Keep the published splits intact. The official archive can contain a
    # small validation/test_seen offer intersection despite its disjointness
    # specification. A separate, offer-disjoint validation copy is therefore
    # created only when explicitly requested. Never remove any test records.
    source_shared = find_offer_overlaps(splits)
    leaked_validation_ids = sorted(
        set(source_shared["validation_vs_test_seen"])
        | set(source_shared["validation_vs_test_unseen"])
    )
    if quarantine_validation_overlap:
        # No train/evaluation intersection is acceptable under this policy.
        for name in ("train_vs_validation", "train_vs_test_seen", "train_vs_test_unseen"):
            if source_shared[name]:
                raise DatasetValidationError(
                    f"quarantine does not permit {name} offer leakage: "
                    f"{len(source_shared[name])} shared IDs"
                )
        validation = splits["validation"]
        keep = ~(
            validation["id_left"].isin(leaked_validation_ids)
            | validation["id_right"].isin(leaked_validation_ids)
        )
        validation_safe = validation.loc[keep].reset_index(drop=True)
        if validation_safe.empty:
            raise DatasetValidationError("quarantine removed all validation pairs")
        audited_splits = dict(splits)
        audited_splits["validation"] = validation_safe
        overlaps = audit_offers_across_splits(audited_splits)
    else:
        # Preserve strict behavior by default; require explicit opt-in for
        # any validation/test overlap. No outputs are written on failure.
        overlaps = audit_offers_across_splits(splits)
        validation_safe = splits["validation"]

    excluded_pairs = len(splits["validation"]) - len(validation_safe)
    source_digest = (
        file_sha256(source)
        if source.is_file()
        else {role: file_sha256(source / member) for role, member in members.items()}
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
                "unique_offers": len(set(df["id_left"].tolist()) | set(df["id_right"].tolist())),
            }
            for role, df in splits.items()
        },
        "offer_overlap_audit": overlaps,
        "source_offer_overlap_counts": {name: len(ids) for name, ids in source_shared.items()},
        "validation_quarantine": {
            "enabled": quarantine_validation_overlap,
            "excluded_offer_ids": leaked_validation_ids if quarantine_validation_overlap else [],
            "excluded_pair_count": excluded_pairs,
            "original_pair_count": len(splits["validation"]),
            "safe_pair_count": len(validation_safe),
            "safe_output": "validation_safe.parquet",
            "selection_basis": "offer IDs only; no test labels or predictions",
        },
        "data_policy": (
            "official splits preserved in separate files; model selection must use "
            "validation_safe.parquet; test labels are excluded from development"
        ),
        "expected_official_counts_checked": enforce_official_counts,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for role, df in splits.items():
        df.to_parquet(output_dir / f"{role}.parquet", index=False, engine="pyarrow")
    validation_safe.to_parquet(
        output_dir / "validation_safe.parquet", index=False, engine="pyarrow"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

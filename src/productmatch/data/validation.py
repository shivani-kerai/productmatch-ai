"""Strict, benchmark-aware validation for WDC Products pair-wise data."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

TEXT_ATTRIBUTES = ("title", "brand", "description", "price", "priceCurrency")
REQUIRED_COLUMNS = (
    "pair_id", "label", "id_left", "id_right", "title_left", "title_right",
)
OPTIONAL_COLUMNS = (
    "is_hard_negative",
    "cluster_id_left",
    "cluster_id_right",
    *(f"{name}_{side}" for name in TEXT_ATTRIBUTES if name != "title"
      for side in ("left", "right")),
)
STANDARD_COLUMNS = (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS)


class DatasetValidationError(ValueError):
    """Raised when a source split fails a data-quality or leakage check."""


def validate_pairs(records: list[Mapping[str, object]], split: str) -> pd.DataFrame:
    """Check schema, labels, identifiers and duplication; retain source row order.

    Cluster IDs and pair IDs are retained for auditing, NEVER as predictive features.
    """
    if not records:
        raise DatasetValidationError(f"{split}: empty split")
    df = pd.DataFrame.from_records(records)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DatasetValidationError(f"{split}: missing required columns: {sorted(missing)}")
    if df["pair_id"].isna().any() or df["pair_id"].astype(str).str.strip().eq("").any():
        raise DatasetValidationError(f"{split}: blank pair_id")

    for column in ("id_left", "id_right"):
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.isna().any() or (numeric % 1 != 0).any() or (numeric < 0).any():
            raise DatasetValidationError(f"{split}: invalid nonnegative integer in {column}")
        df[column] = numeric.astype("int64")

    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise DatasetValidationError(f"{split}: labels must be binary 0/1")
    df["label"] = labels.astype("int64")

    expected = df["id_left"].astype(str) + "#" + df["id_right"].astype(str)
    if not df["pair_id"].astype(str).eq(expected).all():
        raise DatasetValidationError(f"{split}: pair_id does not match id_left#id_right")
    if df["pair_id"].duplicated().any():
        raise DatasetValidationError(f"{split}: duplicate pair_id")
    unordered = pd.Series(
        [tuple(sorted((int(a), int(b)))) for a, b in zip(df["id_left"], df["id_right"])],
        dtype="object",
    )
    if unordered.duplicated().any():
        raise DatasetValidationError(f"{split}: duplicate unordered product pair")
    if (df["id_left"] == df["id_right"]).any():
        raise DatasetValidationError(f"{split}: self-pair found")

    for side in ("left", "right"):
        column = f"title_{side}"
        blank = df[column].isna() | df[column].astype(str).str.strip().eq("")
        if blank.any():
            raise DatasetValidationError(f"{split}: missing/blank {column}")

    for column in (f"{a}_{s}" for a in TEXT_ATTRIBUTES for s in ("left", "right")):
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = df[column].astype("string")

    for column in ("cluster_id_left", "cluster_id_right"):
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = df[column].astype("string")

    if "is_hard_negative" not in df.columns:
        df["is_hard_negative"] = pd.NA
    else:
        valid = df["is_hard_negative"].isin([True, False, 0, 1])
        if not (valid | df["is_hard_negative"].isna()).all():
            raise DatasetValidationError(f"{split}: invalid is_hard_negative flag")
    df["is_hard_negative"] = df["is_hard_negative"].astype("boolean")
    df["pair_id"] = df["pair_id"].astype("string")
    return df.loc[:, list(STANDARD_COLUMNS)].reset_index(drop=True)


def audit_offers_across_splits(splits: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    """Check train/validation against each other and each test variant.

    Alternative test variants can legitimately reuse test offers, so they are
    deliberately not compared against one another.
    """

    def offers(df: pd.DataFrame) -> set[int]:
        return set(df["id_left"].astype(int)) | set(df["id_right"].astype(int))

    offer_sets = {name: offers(df) for name, df in splits.items()}
    checks = (
        ("train", "validation"),
        ("train", "test_seen"),
        ("train", "test_unseen"),
        ("validation", "test_seen"),
        ("validation", "test_unseen"),
    )
    audited: dict[str, int] = {}
    for left, right in checks:
        if left not in offer_sets or right not in offer_sets:
            raise DatasetValidationError(f"missing split for audit: {left} or {right}")
        overlap = len(offer_sets[left] & offer_sets[right])
        audited[f"{left}_vs_{right}"] = overlap
        if overlap:
            raise DatasetValidationError(
                f"offer leakage between {left} and {right}: {overlap} shared offer IDs"
            )
    return audited

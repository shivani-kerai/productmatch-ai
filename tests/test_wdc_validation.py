"""Offline data-integrity tests; never download third-party data in CI."""

import pytest

from productmatch.data.validation import (
    DatasetValidationError,
    audit_offers_across_splits,
    validate_pairs,
)


def pair(left: int, right: int, label: int = 0) -> dict[str, object]:
    return {
        "pair_id": f"{left}#{right}",
        "id_left": left,
        "id_right": right,
        "label": label,
        "title_left": f"Camera model {left}",
        "title_right": f"Camera model {right}",
        "brand_left": "Acme",
        "brand_right": None,
        "is_hard_negative": label == 0,
        "cluster_id_left": 17,
        "cluster_id_right": 17 if label else 18,
    }


def test_valid_official_pair_schema() -> None:
    df = validate_pairs([pair(1, 2, 1), pair(3, 4)], "train")
    assert list(df["label"]) == [1, 0]
    assert df["brand_right"].isna().all()
    assert list(df["pair_id"]) == ["1#2", "3#4"]
    assert str(df["id_left"].dtype) == "int64"


@pytest.mark.parametrize(
    "corrupt",
    [
        {"label": 2},
        {"label": -1},
        {"label": "matching"},
        {"pair_id": "bad"},
        {"id_left": None},
        {"title_left": " "},
        {"id_right": 1},
        {"is_hard_negative": "maybe"},
    ],
)
def test_invalid_records_are_rejected(corrupt: dict[str, object]) -> None:
    record = pair(1, 2)
    record.update(corrupt)
    with pytest.raises(DatasetValidationError):
        validate_pairs([record], "train")


def test_duplicate_or_reversed_pairs_rejected() -> None:
    with pytest.raises(DatasetValidationError, match="duplicate"):
        validate_pairs([pair(1, 2), pair(1, 2)], "train")
    with pytest.raises(DatasetValidationError, match="duplicate"):
        validate_pairs([pair(1, 2), pair(2, 1)], "train")


def test_missing_required_column_rejected() -> None:
    record = pair(1, 2)
    del record["id_right"]
    with pytest.raises(DatasetValidationError, match="missing required"):
        validate_pairs([record], "train")


def test_offer_level_leakage_detected() -> None:
    splits = {
        "train": validate_pairs([pair(1, 2)], "train"),
        "validation": validate_pairs([pair(2, 3)], "validation"),
        "test_seen": validate_pairs([pair(4, 5)], "test_seen"),
        "test_unseen": validate_pairs([pair(6, 7)], "test_unseen"),
    }
    with pytest.raises(DatasetValidationError, match="offer leakage"):
        audit_offers_across_splits(splits)


def test_alternative_tests_may_share_offers() -> None:
    splits = {
        "train": validate_pairs([pair(1, 2)], "train"),
        "validation": validate_pairs([pair(3, 4)], "validation"),
        "test_seen": validate_pairs([pair(5, 6)], "test_seen"),
        "test_unseen": validate_pairs([pair(5, 7)], "test_unseen"),
    }
    assert all(overlap == 0 for overlap in audit_offers_across_splits(splits).values())

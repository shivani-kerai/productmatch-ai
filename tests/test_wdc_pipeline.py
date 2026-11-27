"""End-to-end checks on a tiny synthetic WDC-shaped ZIP, no downloads."""

import gzip
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from productmatch.data.eda import development_summary, profile_split
from productmatch.data.ingestion import (
    SplitSelectionError,
    available_members,
    read_jsonl_member,
    select_split_members,
)
from productmatch.data.pipeline import prepare_dataset
from productmatch.data.validation import DatasetValidationError


def record(left: int, right: int, label: int = 0) -> dict[str, object]:
    return {
        "pair_id": f"{left}#{right}",
        "label": label,
        "id_left": left,
        "id_right": right,
        "title_left": f"Acme Camera {left} 250ML",
        "title_right": f"Acme Camera {right} 500ml",
        "brand_left": "Acme",
        "brand_right": "Acme",
        "description_left": None,
        "description_right": "Camera",
        "price_left": "129.99",
        "price_right": "129.99",
        "priceCurrency_left": "USD",
        "priceCurrency_right": "USD",
        "is_hard_negative": label == 0,
    }


@pytest.fixture
def mini_zip(tmp_path: Path) -> Path:
    files = {
        "training/wdcproducts50cc50rnd000un_train_medium.json.gz": [
            record(1, 2, 1), record(3, 4)
        ],
        "validation/wdcproducts50cc50rnd000un_valid_medium.json.gz": [
            record(5, 6, 1)
        ],
        "test/wdcproducts50cc50rnd000un_test.json.gz": [record(7, 8)],
        "test/wdcproducts50cc50rnd100un_test.json.gz": [record(7, 9)],
    }
    archive = tmp_path / "50pair.zip"
    with ZipFile(archive, "w") as zf:
        for member, rows in files.items():
            payload = "".join(json.dumps(row) + "\n" for row in rows)
            zf.writestr(member, gzip.compress(payload.encode("utf-8")))
    return archive


def test_member_discovery_and_loading(mini_zip: Path) -> None:
    members = available_members(mini_zip)
    chosen = select_split_members(members)
    assert len(chosen) == 4
    assert len(read_jsonl_member(mini_zip, chosen["train"])) == 2


def test_explicit_member_override_and_missing_member(mini_zip: Path) -> None:
    members = available_members(mini_zip)
    train = select_split_members(members)["train"]
    assert select_split_members(members, {"train": train})["train"] == train
    with pytest.raises(SplitSelectionError, match="not in archive"):
        select_split_members(members, {"train": "bad.json.gz"})


def test_missing_role_or_ambiguous_role_fails_closed() -> None:
    with pytest.raises(SplitSelectionError, match="cannot uniquely select"):
        select_split_members(["train_medium.json.gz"])


def test_pipeline_produces_reproducible_official_splits(
    mini_zip: Path, tmp_path: Path
) -> None:
    target = tmp_path / "processed"
    manifest = prepare_dataset(
        mini_zip, target, enforce_official_counts=False
    )
    assert manifest["splits"]["train"]["rows"] == 2
    assert manifest["splits"]["validation"]["rows"] == 1
    assert manifest["offer_overlap_audit"]["train_vs_validation"] == 0
    train = pd.read_parquet(target / "train.parquet")
    assert list(train["pair_id"]) == ["1#2", "3#4"]
    assert train.iloc[0]["title_left_normalized"] == "acme camera 1 250 ml"
    assert (target / "test_seen.parquet").exists()
    assert (target / "test_unseen.parquet").exists()
    assert json.loads((target / "manifest.json").read_text()) == manifest
    assert prepare_dataset(mini_zip, target, enforce_official_counts=False) == manifest


def test_official_counts_are_checked_by_default(mini_zip: Path, tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="expected 6000"):
        prepare_dataset(mini_zip, tmp_path / "processed")


def test_eda_excludes_tests(mini_zip: Path, tmp_path: Path) -> None:
    target = tmp_path / "processed"
    prepare_dataset(mini_zip, target, enforce_official_counts=False)
    train = pd.read_parquet(target / "train.parquet")
    validation = pd.read_parquet(target / "validation.parquet")
    report = development_summary({"train": train, "validation": validation})
    assert report["train"]["positive"] == 1
    assert report["train"]["negative"] == 1
    assert report["train"]["attribute_missing_fraction"]["description_left"] == 1.0
    assert profile_split(validation)["positive"] == 1
    with pytest.raises(ValueError, match="exactly train and validation"):
        development_summary({"train": train})


def test_source_validation_overlap_requires_explicit_quarantine(
    tmp_path: Path,
) -> None:
    """Replicate real WDC symptom without touching official source splits."""
    archive = tmp_path / "contaminated.zip"
    members = {
        "training/wdcproducts50cc50rnd000un_train_medium.json.gz": [
            record(1, 2, 1),
        ],
        "validation/wdcproducts50cc50rnd000un_valid_medium.json.gz": [
            record(3, 4, 1), record(5, 6, 0),
        ],
        "test/wdcproducts50cc50rnd000un_gs.json.gz": [
            record(3, 7, 0),
        ],
        "test/wdcproducts50cc50rnd100un_gs.json.gz": [
            record(8, 9, 1),
        ],
    }
    with ZipFile(archive, "w") as zf:
        for member, rows in members.items():
            payload = "".join(json.dumps(row) + "\n" for row in rows)
            zf.writestr(member, gzip.compress(payload.encode("utf-8")))

    destination = tmp_path / "processed"
    with pytest.raises(DatasetValidationError, match="validation and test_seen"):
        prepare_dataset(archive, destination, enforce_official_counts=False)
    assert not (destination / "manifest.json").exists()

    manifest = prepare_dataset(
        archive,
        destination,
        enforce_official_counts=False,
        quarantine_validation_overlap=True,
    )
    assert manifest["source_offer_overlap_counts"]["validation_vs_test_seen"] == 1
    assert manifest["offer_overlap_audit"]["validation_vs_test_seen"] == 0
    assert manifest["validation_quarantine"]["excluded_offer_ids"] == [3]
    assert manifest["validation_quarantine"]["excluded_pair_count"] == 1
    assert manifest["validation_quarantine"]["safe_pair_count"] == 1
    assert len(pd.read_parquet(destination / "validation.parquet")) == 2
    assert list(pd.read_parquet(destination / "validation_safe.parquet")["pair_id"]) == [
        "5#6"
    ]
    assert list(pd.read_parquet(destination / "test_seen.parquet")["pair_id"]) == [
        "3#7"
    ]

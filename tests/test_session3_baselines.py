"""Session 3 offline fixture tests: leakage, deterministic features and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from productmatch.evaluation.classification import (
    choose_f1_threshold,
    classification_metrics,
    error_analysis,
)
from productmatch.features.pairwise import (
    FEATURE_NAMES,
    build_pairwise_features,
    pair_features,
)
from productmatch.models.baselines import MODEL_CHOICES, make_estimator
from productmatch.models.training import _load_development, train_baselines


def pair(left: int, right: int, label: int) -> dict[str, object]:
    same = label == 1
    return {
        "pair_id": f"{left}#{right}",
        "id_left": left,
        "id_right": right,
        "label": label,
        "title_left": f"Acme Camera X200 {250 if same else 250}ml",
        "title_right": f"Acme Camera {'X200' if same else 'Y300'} "
        f"{250 if same else 500}ml",
        "brand_left": "Acme",
        "brand_right": "Acme" if same else "Generic",
        "description_left": "Digital camera",
        "description_right": "Digital camera" if same else "Accessory",
        "price_left": "100.0",
        "price_right": "100.0" if same else "180.0",
        "priceCurrency_left": "USD",
        "priceCurrency_right": "USD",
        "is_hard_negative": not same,
        "cluster_id_left": 777 if same else 888,
        "cluster_id_right": 777 if same else 999,
    }


def test_features_do_not_use_identity_or_label() -> None:
    row = pair(1, 2, 1)
    original = pair_features(row)
    row.update({
        "label": 0,
        "pair_id": "injected-pair-id",
        "cluster_id_left": 999999,
        "cluster_id_right": -1,
        "is_hard_negative": True,
        "id_left": 900,
        "id_right": 901,
    })
    assert pair_features(row) == original
    assert len(original) == len(FEATURE_NAMES)
    assert all(np.isfinite(list(original.values())))


def test_quantity_and_price_conflicts_are_observable() -> None:
    same = pair_features({
        "title_left": "Acme Shampoo 250 ml",
        "title_right": "Acme Shampoo 0.25 l",
        "price_left": "100",
        "price_right": "100",
        "priceCurrency_left": "USD",
        "priceCurrency_right": "USD",
    })
    different = pair_features({
        "title_left": "Acme Shampoo 250ml pack of 2",
        "title_right": "Acme Shampoo 500ml pack of 3",
        "price_left": "100",
        "price_right": "200",
        "priceCurrency_left": "USD",
        "priceCurrency_right": "INR",
    })
    assert same["quantity_match"] == 1.0
    assert same["price_comparable"] == 1.0
    assert different["quantity_conflict"] == 1.0
    assert different["pack_count_conflict"] == 1.0
    assert different["currency_conflict"] == 1.0
    assert different["price_comparable"] == 0.0


def test_features_have_deterministic_order_and_values() -> None:
    df = pd.DataFrame([pair(1, 2, 1), pair(3, 4, 0)])
    a = build_pairwise_features(df)
    b = build_pairwise_features(df)
    assert list(a.columns) == list(FEATURE_NAMES)
    pd.testing.assert_frame_equal(a, b)
    assert not a.isna().any().any()


def test_f1_threshold_uses_validation_scores_only() -> None:
    truth = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.3, 0.7, 0.9])
    threshold = choose_f1_threshold(truth, probabilities)
    assert 0.3 < threshold <= 0.7
    result = classification_metrics(truth, probabilities, threshold)
    assert result["f1"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["tp"] == 2


def test_reject_malformed_or_single_class_scores() -> None:
    with pytest.raises(ValueError, match="both classes"):
        choose_f1_threshold(np.array([0, 0]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="between 0 and 1"):
        classification_metrics(
            np.array([0, 1]), np.array([-0.1, 0.8])
        )


def test_error_analysis_masks_only_actual_mistakes() -> None:
    data = pd.DataFrame([pair(1, 2, 0), pair(3, 4, 1)])
    errors, hard = error_analysis(data, np.array([0.9, 0.2]), 0.5)
    assert set(errors["error_type"]) == {"false_positive", "false_negative"}
    assert hard["hard_negative_pairs"] == 1
    assert hard["hard_negative_false_positives"] == 1


@pytest.fixture
def tiny_development(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    training = pd.DataFrame(
        [pair(i, i + 1, (i // 2) % 2) for i in range(0, 32, 2)]
    )
    validation = pd.DataFrame(
        [pair(i, i + 1, (i - 100) % 2) for i in range(100, 116)]
    )
    training.to_parquet(root / "train.parquet", index=False)
    validation.to_parquet(root / "validation_safe.parquet", index=False)
    # Deliberately write no test Parquet files.
    manifest = {
        "offer_overlap_audit": {
            "train_vs_validation": 0,
            "train_vs_test_seen": 0,
            "train_vs_test_unseen": 0,
            "validation_vs_test_seen": 0,
            "validation_vs_test_unseen": 0,
        },
        "validation_quarantine": {
            "safe_output": "validation_safe.parquet",
            "safe_pair_count": len(validation),
            "excluded_pair_count": 2,
        },
        "splits": {"train": {"rows": len(training)}},
        "archive_sha256": "synthetic",
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_training_refuses_original_validation(tiny_development: Path) -> None:
    (tiny_development / "validation_safe.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="validation_safe"):
        _load_development(tiny_development)


def test_training_rejects_offer_reuse(tmp_path: Path, tiny_development: Path) -> None:
    valid = pd.read_parquet(tiny_development / "validation_safe.parquet")
    valid.loc[0, "id_left"] = 0
    valid.to_parquet(tiny_development / "validation_safe.parquet", index=False)
    with pytest.raises(ValueError, match="share offer IDs"):
        _load_development(tiny_development)


@pytest.mark.parametrize("model", MODEL_CHOICES)
def test_real_estimators_train_offline(
    tiny_development: Path, tmp_path: Path, model: str
) -> None:
    result = train_baselines(
        tiny_development,
        tmp_path / "reports",
        tmp_path / "artifacts",
        model_names=[model],
    )
    assert result["train_pairs"] == 16
    assert result["validation_pairs"] == 16
    assert result["source_validation_excluded_pairs"] == 2
    assert result["best_validation_model_by_ap"] == model
    assert (tmp_path / "reports" / "metrics.json").exists()
    assert (tmp_path / "artifacts" / f"{model}.joblib").exists()
    assert 0 <= result["models"][model]["default_threshold_metrics"]["f1"] <= 1


def test_invalid_model_name(tiny_development: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid"):
        train_baselines(
            tiny_development, tmp_path / "reports", tmp_path / "artifacts",
            model_names=["not-a-model"]
        )


def test_model_factory_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        make_estimator("not-a-model")

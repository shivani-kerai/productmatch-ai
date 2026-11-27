"""Reproducible training and comparison on WDC train + leakage-safe validation.

No official test file is loaded. Threshold tuning and model selection happen
on the same validation data, so tuned metrics are development estimates only.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from productmatch.evaluation.classification import (
    METRIC_NOTE,
    choose_f1_threshold,
    classification_metrics,
    error_analysis,
)
from productmatch.features.pairwise import FEATURE_NAMES, build_pairwise_features
from productmatch.models.baselines import DEFAULT_SEED, MODEL_CHOICES, make_estimator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_development(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fail closed when the leakage-safe validation copy is absent or inconsistent."""
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepare WDC data first; missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_path = data_dir / "validation_safe.parquet"
    if not validation_path.is_file():
        raise FileNotFoundError(
            f"{validation_path} is required; never tune on validation.parquet. "
            "Run prepare_data.py --quarantine-validation-overlap."
        )
    audit = manifest.get("offer_overlap_audit", {})
    if not audit or any(n != 0 for n in audit.values()):
        raise ValueError("manifest does not record a clean development/evaluation overlap audit")

    original = manifest.get("validation_quarantine", {})
    if original.get("safe_output") != "validation_safe.parquet":
        raise ValueError("validation provenance manifest does not identify the safe split")
    train = pd.read_parquet(data_dir / "train.parquet")
    validation = pd.read_parquet(validation_path)
    if not train.shape[0] or not validation.shape[0]:
        raise ValueError("train and validation must be nonempty")
    if len(validation) != original.get("safe_pair_count"):
        raise ValueError("validation_safe size differs from the provenance manifest")
    if len(train) != manifest.get("splits", {}).get("train", {}).get("rows"):
        raise ValueError("train size differs from the provenance manifest")
    for name, frame in (("train", train), ("validation_safe", validation)):
        if "label" not in frame:
            raise ValueError(f"{name}: missing label")
        labels = frame["label"].to_numpy(dtype=int)
        if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
            raise ValueError(f"{name}: binary labels must include both classes")

    train_offers = set(train["id_left"]) | set(train["id_right"])
    valid_offers = set(validation["id_left"]) | set(validation["id_right"])
    if train_offers & valid_offers:
        raise ValueError("train and validation_safe share offer IDs")
    return train, validation, manifest


def _feature_rank(model: object, model_name: str) -> list[dict[str, float | str]]:
    if model_name == "logistic_regression":
        weights = model.named_steps["model"].coef_[0]
    else:
        weights = model.feature_importances_
    return sorted(
        [
            {"feature": name, "importance": round(float(weight), 8)}
            for name, weight in zip(FEATURE_NAMES, weights, strict=True)
        ],
        key=lambda item: abs(item["importance"]),
        reverse=True,
    )


def train_baselines(
    data_dir: Path,
    report_dir: Path,
    artifact_dir: Path,
    model_names: Sequence[str] = MODEL_CHOICES,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Train baseline models; save local artifacts and validation error reports."""
    invalid_models = sorted(set(model_names) - set(MODEL_CHOICES))
    if invalid_models or not model_names:
        raise ValueError(f"models must be one or more of {MODEL_CHOICES}; invalid: {invalid_models}")

    data_dir = Path(data_dir)
    train, validation, manifest = _load_development(data_dir)
    x_train = build_pairwise_features(train)
    x_validation = build_pairwise_features(validation)
    y_train = train["label"].to_numpy(dtype=int)
    y_validation = validation["label"].to_numpy(dtype=int)

    report_dir = Path(report_dir)
    artifact_dir = Path(artifact_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "session": 3,
        "task": "WDC pairwise product matching",
        "model_selection_split": "validation_safe.parquet",
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "source_validation_excluded_pairs": manifest["validation_quarantine"][
            "excluded_pair_count"
        ],
        "source_archive_sha256": manifest.get("archive_sha256"),
        "input_sha256": {
            "train.parquet": _sha256(data_dir / "train.parquet"),
            "validation_safe.parquet": _sha256(data_dir / "validation_safe.parquet"),
            "manifest.json": _sha256(data_dir / "manifest.json"),
        },
        "feature_names": list(FEATURE_NAMES),
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "evaluation_note": METRIC_NOTE,
        "models": {},
    }

    for model_name in dict.fromkeys(model_names):
        estimator = make_estimator(model_name, seed=seed)
        estimator.fit(x_train, y_train)
        probabilities = estimator.predict_proba(x_validation)[:, 1]
        cutoff = choose_f1_threshold(y_validation, probabilities)

        default_metrics = classification_metrics(y_validation, probabilities, 0.5)
        tuned_metrics = classification_metrics(y_validation, probabilities, cutoff)
        errors, hard = error_analysis(validation, probabilities, cutoff)

        model_path = artifact_dir / f"{model_name}.joblib"
        joblib.dump(
            {
                "model": estimator,
                "feature_names": FEATURE_NAMES,
                "decision_threshold": cutoff,
                "model_name": model_name,
                "seed": seed,
                "train_data_sha256": result["input_sha256"]["train.parquet"],
            },
            model_path,
        )
        errors.to_csv(report_dir / f"{model_name}_errors.csv", index=False)
        result["models"][model_name] = {
            "default_threshold_metrics": default_metrics,
            "validation_tuned_threshold_metrics": tuned_metrics,
            "hard_negative_analysis": hard,
            "feature_importance": _feature_rank(estimator, model_name),
            "errors_csv": str(report_dir / f"{model_name}_errors.csv"),
            "model_artifact": str(model_path),
        }

    # AP is a threshold-independent *validation selection criterion*, not
    # an estimate of unseen test generalization or deployment performance.
    ranks = [
        (
            name,
            model_result["default_threshold_metrics"]["average_precision"],
        )
        for name, model_result in result["models"].items()
    ]
    result["best_validation_model_by_ap"] = max(
        ranks, key=lambda item: (item[1], item[0])
    )[0]
    (report_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result

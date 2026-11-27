"""Fit a model-bound sigmoid calibrator on offer-disjoint development rows.

Model selection already used the full validation_safe set in Session 3, so
the audit partition is not an independent, unbiased model-selection test.
This script does NOT open the official WDC seen or unseen test variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from productmatch.api.runtime import load_model
from productmatch.evaluation.calibration import (
    apply_sigmoid_calibration,
    file_sha256,
    probability_logits,
    reliability_metrics,
    split_for_calibration,
)
from productmatch.evaluation.classification import (
    choose_f1_threshold,
    classification_metrics,
)
from productmatch.features.pairwise import FEATURE_NAMES, build_pairwise_features
from productmatch.models.training import _load_development


def fit_calibration(
    *,
    data_dir: Path,
    model_path: Path,
    output_path: Path,
    report_path: Path,
    seed: int = 42,
    audit_fraction: float = 0.3,
) -> dict[str, object]:
    """Calibrate a trusted frozen model without fitting or changing that model."""
    _, validation, manifest = _load_development(data_dir)
    model, _, model_name = load_model(model_path)
    raw = np.asarray(
        model.predict_proba(build_pairwise_features(validation))[:, 1], dtype=float
    )
    train_idx, audit_idx = split_for_calibration(
        validation, seed=seed, audit_fraction=audit_fraction
    )
    labels = validation["label"].to_numpy(dtype=int)
    calibration_labels = labels[train_idx]
    diagnostic_labels = labels[audit_idx]

    estimator = LogisticRegression(
        solver="lbfgs", random_state=seed, max_iter=2000
    )
    estimator.fit(probability_logits(raw[train_idx]), calibration_labels)
    tuned = apply_sigmoid_calibration(estimator, raw[train_idx])
    cutoff = choose_f1_threshold(calibration_labels, tuned)

    diagnostic_raw = raw[audit_idx]
    diagnostic_calibrated = apply_sigmoid_calibration(estimator, diagnostic_raw)
    model_digest = file_sha256(model_path)
    safe_digest = file_sha256(Path(data_dir) / "validation_safe.parquet")
    result: dict[str, object] = {
        "session": 6,
        "calibration_type": "sigmoid_logit",
        "model_name": model_name,
        "model_sha256": model_digest,
        "validation_safe_sha256": safe_digest,
        "model_selection_warning": (
            "Session 3 previously selected the model and threshold using "
            "validation_safe; these offer-disjoint calibration/audit partitions "
            "are secondary development diagnostics, not unbiased held-out test metrics."
        ),
        "source_validation_overlap_exclusions": manifest["validation_quarantine"][
            "excluded_pair_count"
        ],
        "official_test_sets_read": False,
        "calibration_pairs": int(len(train_idx)),
        "audit_pairs": int(len(audit_idx)),
        "selected_threshold": float(cutoff),
        "seed": seed,
        "audit_fraction": audit_fraction,
        "audit_raw_reliability": reliability_metrics(diagnostic_labels, diagnostic_raw),
        "audit_calibrated_reliability": reliability_metrics(
            diagnostic_labels, diagnostic_calibrated
        ),
        "audit_raw_at_0_5": classification_metrics(
            diagnostic_labels, diagnostic_raw, 0.5
        ),
        "audit_calibrated_at_selected_threshold": classification_metrics(
            diagnostic_labels, diagnostic_calibrated, cutoff
        ),
    }
    payload = {
        "schema_version": 1,
        "kind": "sigmoid_logit",
        "model_sha256": model_digest,
        "feature_names": tuple(FEATURE_NAMES),
        "estimator": estimator,
        "decision_threshold": float(cutoff),
        "source_validation_safe_sha256": safe_digest,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/wdc50_medium")
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/session3_baselines/logistic_regression.joblib"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/session6_calibration/sigmoid.joblib"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/session6_calibration/reliability.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-fraction", type=float, default=0.3)
    args = parser.parse_args()
    report = fit_calibration(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_path=args.output,
        report_path=args.report,
        seed=args.seed,
        audit_fraction=args.audit_fraction,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

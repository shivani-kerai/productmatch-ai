"""Fit a model-bound sigmoid calibrator on leakage-safe development rows.

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
    offer_component_groups,
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
    development_only: bool = False,
) -> dict[str, object]:
    """Calibrate a trusted frozen model without fitting or changing that model."""
    _, validation, manifest = _load_development(data_dir)
    model, _, model_name = load_model(model_path)
    raw = np.asarray(model.predict_proba(build_pairwise_features(validation))[:, 1], dtype=float)
    # Explicit opt-in: when offer graph components cannot be separated, fit
    # only a development calibrator, with *no* independent audit metrics.
    # This never reads either official test file or original validation.parquet.
    if development_only:
        train_idx = np.arange(len(validation))
        audit_idx = np.array([], dtype=int)
    else:
        train_idx, audit_idx = split_for_calibration(
            validation, seed=seed, audit_fraction=audit_fraction
        )

    labels = validation["label"].to_numpy(dtype=int)
    calibration_labels = labels[train_idx]
    estimator = LogisticRegression(solver="lbfgs", random_state=seed, max_iter=2000)
    estimator.fit(probability_logits(raw[train_idx]), calibration_labels)
    tuned = apply_sigmoid_calibration(estimator, raw[train_idx])
    cutoff = choose_f1_threshold(calibration_labels, tuned)

    # Fitted-data metrics are resubstitution diagnostics, never model quality claims.
    fit_raw = reliability_metrics(calibration_labels, raw[train_idx])
    fit_calibrated = reliability_metrics(calibration_labels, tuned)
    diagnostic_labels = labels[audit_idx]
    diagnostic_raw = raw[audit_idx]
    diagnostic_calibrated = (
        apply_sigmoid_calibration(estimator, diagnostic_raw)
        if len(audit_idx)
        else np.array([], dtype=float)
    )
    audit_raw = (
        reliability_metrics(diagnostic_labels, diagnostic_raw) if len(audit_idx) else None
    )
    audit_calibrated = (
        reliability_metrics(diagnostic_labels, diagnostic_calibrated)
        if len(audit_idx)
        else None
    )
    audit_raw_metrics = (
        classification_metrics(diagnostic_labels, diagnostic_raw, 0.5)
        if len(audit_idx)
        else None
    )
    audit_tuned_metrics = (
        classification_metrics(diagnostic_labels, diagnostic_calibrated, cutoff)
        if len(audit_idx)
        else None
    )
    model_digest = file_sha256(model_path)
    safe_digest = file_sha256(Path(data_dir) / "validation_safe.parquet")
    result: dict[str, object] = {
        "session": 6,
        "calibration_type": "sigmoid_logit",
        "model_name": model_name,
        "model_sha256": model_digest,
        "validation_safe_sha256": safe_digest,
        "model_selection_warning": (
            "Session 3 already selected the model using validation_safe. "
            "Development-only mode reuses all validation rows for fitting, "
            "and has NO audit set, independent metrics or generalization claim."
            if development_only
            else "Session 3 previously selected the model using validation_safe; "
            "even offer-disjoint calibration/audit partitions here are secondary "
            "development diagnostics, not unbiased held-out test metrics."
        ),
        "calibration_policy": (
            "development_only_full_validation"
            if development_only
            else "offer_disjoint_development_audit"
        ),
        "audit_available": not development_only,
        "independent_offer_components": int(len(np.unique(offer_component_groups(validation)))),
        "fit_raw_reliability_in_sample": fit_raw,
        "fit_calibrated_reliability_in_sample": fit_calibrated,
        "source_validation_overlap_exclusions": manifest["validation_quarantine"][
            "excluded_pair_count"
        ],
        "official_test_sets_read": False,
        "calibration_pairs": int(len(train_idx)),
        "audit_pairs": int(len(audit_idx)),
        "selected_threshold": float(cutoff),
        "seed": seed,
        "audit_fraction": audit_fraction,
        "audit_raw_reliability": audit_raw,
        "audit_calibrated_reliability": audit_calibrated,
        "audit_raw_at_0_5": audit_raw_metrics,
        "audit_calibrated_at_selected_threshold": audit_tuned_metrics,
    }
    payload = {
        "schema_version": 1,
        "kind": "sigmoid_logit",
        "model_sha256": model_digest,
        "feature_names": tuple(FEATURE_NAMES),
        "estimator": estimator,
        "decision_threshold": float(cutoff),
        "source_validation_safe_sha256": safe_digest,
        "calibration_policy": result["calibration_policy"],
        "audit_available": not development_only,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/wdc50_medium"))
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
    parser.add_argument(
        "--development-only",
        action="store_true",
        help=(
            "Opt into fitting on all validation_safe pairs when offer-disjoint "
            "calibration/audit splitting is impossible. No audit metrics are "
            "produced; results must not be presented as independent evaluation."
        ),
    )
    args = parser.parse_args()
    report = fit_calibration(
        data_dir=args.data_dir,
        model_path=args.model_path,
        output_path=args.output,
        report_path=args.report,
        seed=args.seed,
        audit_fraction=args.audit_fraction,
        development_only=args.development_only,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

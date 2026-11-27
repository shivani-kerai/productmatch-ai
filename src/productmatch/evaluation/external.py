"""External pairwise holdout evaluation with disjoint-offer evidence gates.

No training, threshold optimization or calibration fitting occurs here. External
labels are provided by the operator; automation can audit IDs and metadata but
cannot verify that humans labeled the data correctly or independently.
The official WDC seen and unseen test sets are never read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from productmatch.api.runtime import load_calibration, load_model, match_scores
from productmatch.evaluation.calibration import (
    apply_sigmoid_calibration,
    file_sha256,
    reliability_metrics,
)
from productmatch.evaluation.classification import classification_metrics
from productmatch.retrieval.benchmark import Offer

REQUIRED_COLUMNS = frozenset({"id_left", "id_right", "title_left", "title_right", "label"})
ORIGIN_KINDS = frozenset({"human_reviewed", "synthetic_demo"})


def validate_provenance(manifest: dict[str, object], *, allow_demo: bool) -> dict[str, object]:
    """Check a user-submitted annotation statement; never call it verified."""
    origin = manifest.get("label_origin")
    if origin not in ORIGIN_KINDS:
        raise ValueError("provenance label_origin must be human_reviewed or synthetic_demo")
    if origin == "synthetic_demo" and not allow_demo:
        raise ValueError("synthetic demo labels require explicit --allow-demo")
    if not isinstance(manifest.get("annotation_protocol"), str) or not manifest[
        "annotation_protocol"
    ].strip():
        raise ValueError("provenance must describe the annotation_protocol")
    if not isinstance(manifest.get("source"), str) or not manifest["source"].strip():
        raise ValueError("provenance must identify source")
    labelers = manifest.get("labeler_count")
    if type(labelers) is not int or labelers < 1:
        raise ValueError("provenance labeler_count must be a positive integer")
    return {
        "label_origin": origin,
        "source": str(manifest["source"]),
        "annotation_protocol": str(manifest["annotation_protocol"]),
        "labeler_count": labelers,
        "independence_statement": str(manifest.get("independence_statement", "")),
        "verification_status": "operator_asserted_not_independently_verified",
    }


def _id_set(frame: pd.DataFrame) -> set[str]:
    return set(frame["id_left"].astype(str)) | set(frame["id_right"].astype(str))


def audit_holdout(
    pairs: pd.DataFrame, development_dir: Path
) -> dict[str, int | str]:
    """Fail on reused offer identities, duplicate/reversed pairs or malformed labels."""
    missing = REQUIRED_COLUMNS - set(pairs)
    if missing:
        raise ValueError(f"external pairs missing columns: {sorted(missing)}")
    if len(pairs) < 2 or pairs[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("external evaluation needs at least two complete labeled pairs")
    for column in ("id_left", "id_right", "title_left", "title_right"):
        if pairs[column].str.strip().eq("").any():
            raise ValueError(f"external evaluation contains empty {column}")
    labels = pairs["label"].astype(str).to_numpy()
    if not set(labels) <= {"0", "1"} or len(set(labels)) != 2:
        raise ValueError("external labels must contain both 0 and 1")

    directed = [
        tuple(sorted((str(left), str(right))))
        for left, right in zip(pairs["id_left"], pairs["id_right"], strict=True)
    ]
    if len(set(directed)) != len(directed):
        raise ValueError("duplicate or reversed duplicate external offer pairs")
    if any(left == right for left, right in directed):
        raise ValueError("self-pairs cannot be independently evaluated")

    # Only development files; no official test file is read, even indirectly.
    train_path = Path(development_dir) / "train.parquet"
    validation_path = Path(development_dir) / "validation_safe.parquet"
    train = pd.read_parquet(train_path, columns=["id_left", "id_right"])
    validation = pd.read_parquet(validation_path, columns=["id_left", "id_right"])
    external_ids = _id_set(pairs)
    train_overlap = external_ids & _id_set(train)
    valid_overlap = external_ids & _id_set(validation)
    if train_overlap or valid_overlap:
        raise ValueError(
            "external benchmark contains offer IDs reused in train or validation_safe: "
            f"train={len(train_overlap)}, validation={len(valid_overlap)}"
        )
    return {
        "external_pairs": len(pairs),
        "external_unique_offers": len(external_ids),
        "train_offer_overlap": 0,
        "validation_offer_overlap": 0,
        "identity_audit": "passed_offer_id_disjointness_only",
    }


def _offer_from_row(row: dict[str, object], suffix: str) -> Offer:
    def attribute(key: str) -> str:
        value = row.get(f"{key}_{suffix}")
        return "" if value is None or pd.isna(value) else str(value)

    return Offer(
        offer_id=attribute("id"),
        title=attribute("title"),
        brand=attribute("brand"),
        description=attribute("description"),
        price=attribute("price"),
        currency=attribute("priceCurrency"),
    )


def evaluate_external_pairs(
    *,
    pairs_csv: Path,
    provenance_json: Path,
    development_dir: Path,
    model_path: Path,
    report_path: Path,
    calibration_path: Path | None = None,
    allow_demo: bool = False,
) -> dict[str, object]:
    """Evaluate frozen thresholds on new labeled data, never fit on holdout.

    Label quality and genuinely independent data sourcing must be checked by
    people; this program verifies only required fields and offer ID overlap.
    """
    provenance = validate_provenance(
        json.loads(Path(provenance_json).read_text(encoding="utf-8")),
        allow_demo=allow_demo,
    )
    pairs = pd.read_csv(pairs_csv, dtype="string")
    audit = audit_holdout(pairs, development_dir)
    model, raw_threshold, model_name = load_model(Path(model_path))
    rows = pairs.to_dict(orient="records")
    model_pairs = [
        (_offer_from_row(row, "left"), _offer_from_row(row, "right"))
        for row in rows
    ]
    probabilities = np.asarray(match_scores(model, model_pairs), dtype=float)
    labels = pairs["label"].astype(int).to_numpy()
    raw_metrics = classification_metrics(labels, probabilities, raw_threshold)
    results: dict[str, object] = {
        "raw": {
            "classification": raw_metrics,
            "reliability": reliability_metrics(labels, probabilities),
            "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
            "threshold_source": "frozen_session3_artifact",
        }
    }
    calibration_policy: str | None = None
    calibration_sha: str | None = None
    if calibration_path is not None:
        calibrator, threshold, calibration_policy = load_calibration(
            Path(calibration_path), Path(model_path)
        )
        calibrated = apply_sigmoid_calibration(calibrator, probabilities)
        calibration_sha = file_sha256(Path(calibration_path))
        results["calibrated"] = {
            "classification": classification_metrics(labels, calibrated, threshold),
            "reliability": reliability_metrics(labels, calibrated),
            "roc_auc": round(float(roc_auc_score(labels, calibrated)), 6),
            "threshold_source": "frozen_calibration_artifact",
            "calibration_policy": calibration_policy,
        }

    result: dict[str, object] = {
        "schema_version": 1,
        "session": 7,
        "evaluation_type": "independent_pairwise" if provenance[
            "label_origin"
        ] == "human_reviewed" else "synthetic_demo",
        "verification_status": provenance["verification_status"],
        "no_fitting_or_threshold_tuning_on_external_pairs": True,
        "official_wdc_test_sets_read": False,
        "provenance": provenance,
        "audit": audit,
        "model_name": model_name,
        "model_sha256": file_sha256(Path(model_path)),
        "calibration_sha256": calibration_sha,
        "calibration_policy": calibration_policy,
        "external_pairs_sha256": file_sha256(Path(pairs_csv)),
        "provenance_sha256": file_sha256(Path(provenance_json)),
        "metrics": results,
        "limitations": (
            "Automated offer-ID disjointness does not prove independent "
            "collection or annotation quality. External results are conditional "
            "on operator-provided provenance. Synthetic demo numbers MUST NOT "
            "be cited as real benchmark performance."
        ),
    }
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

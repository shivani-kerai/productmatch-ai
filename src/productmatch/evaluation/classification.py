"""Pairwise classification metrics with explicit validation threshold tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)

METRIC_NAME = "average_precision"
METRIC_NOTE = (
    "Average precision (AP) is the non-interpolated summary of the precision-recall "
    "curve; it is not trapezoidal PR-AUC. Tuned-threshold metrics are measured on "
    "the same validation data and are optimistic; test splits are untouched."
)


def validate_scores(y_true: np.ndarray, probabilities: np.ndarray) -> None:
    if y_true.ndim != 1 or probabilities.ndim != 1 or len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must be aligned 1D arrays")
    if len(y_true) == 0 or not np.isin(y_true, [0, 1]).all():
        raise ValueError("labels must be a nonempty binary 0/1 vector")
    if len(np.unique(y_true)) != 2:
        raise ValueError("both classes are needed for baseline comparison")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must lie between 0 and 1")


def choose_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Optimize F1 on development validation; prefer higher cutoff on ties."""
    validate_scores(y_true, probabilities)
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(precision[:-1]),
        where=(precision[:-1] + recall[:-1]) > 0,
    )
    best = np.flatnonzero(np.isclose(f1, f1.max()))
    return float(thresholds[best[-1]])


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float | int]:
    """Positive-label binary metrics and confusion counts at a fixed cutoff."""
    validate_scores(y_true, probabilities)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie between 0 and 1")
    predicted = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predicted, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 8),
        "pairs": int(len(y_true)),
        "positives": int(sum(y_true)),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def error_analysis(
    validation: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Return local error pairs and simple hard-negative diagnostics.

    Both data and metrics stay local (gitignored). IDs are for auditing only.
    """
    labels = validation["label"].to_numpy(dtype=int)
    validate_scores(labels, probabilities)
    if len(validation) != len(probabilities):
        raise ValueError("validation rows and scores are not aligned")

    preds = (probabilities >= threshold).astype(int)
    incorrect = labels != preds
    cols = [
        name for name in (
            "pair_id", "title_left", "title_right", "brand_left", "brand_right",
            "is_hard_negative",
        ) if name in validation
    ]
    errors = validation.loc[incorrect, cols].copy()
    errors["true_label"] = labels[incorrect]
    errors["predicted_label"] = preds[incorrect]
    errors["match_probability"] = probabilities[incorrect]
    errors["error_type"] = np.where(
        errors["predicted_label"] == 1, "false_positive", "false_negative"
    )
    # The confidence ranking helps inspect the highest-risk mistakes.
    errors["error_confidence"] = np.where(
        errors["predicted_label"] == 1,
        errors["match_probability"],
        1 - errors["match_probability"],
    )
    errors = errors.sort_values("error_confidence", ascending=False).reset_index(drop=True)
    hard: dict[str, float | int | None] = {
        "hard_negative_pairs": 0,
        "hard_negative_false_positives": 0,
        "hard_negative_false_positive_rate": None,
    }
    if "is_hard_negative" in validation:
        flagged = validation["is_hard_negative"].fillna(False).astype(bool).to_numpy()
        hard_mask = (labels == 0) & flagged
        total = int(hard_mask.sum())
        false_matches = int((hard_mask & (preds == 1)).sum())
        hard = {
            "hard_negative_pairs": total,
            "hard_negative_false_positives": false_matches,
            "hard_negative_false_positive_rate": (
                round(false_matches / total, 6) if total else None
            ),
        }
    return errors, hard

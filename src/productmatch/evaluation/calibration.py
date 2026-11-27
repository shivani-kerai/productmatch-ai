"""Offer-aware development score calibration and reliability diagnostics.

Official WDC test variants are never accessed. Warning: Session 3 already
selected models using validation_safe; this split is therefore a secondary
development diagnostic, NOT a truly independent model-selection holdout.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def file_sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def offer_component_groups(pairs: pd.DataFrame) -> np.ndarray:
    """Union offers joined by *any* pair; a component cannot cross the split."""
    if not {"id_left", "id_right"}.issubset(pairs):
        raise ValueError("id_left and id_right are required for grouped splitting")
    parent: dict[int, int] = {}

    def find(value: int) -> int:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    left = pairs["id_left"].astype(int).tolist()
    right = pairs["id_right"].astype(int).tolist()
    for a, b in zip(left, right, strict=True):
        union(a, b)
    return np.asarray([find(a) for a in left], dtype=np.int64)


def split_for_calibration(
    pairs: pd.DataFrame,
    *,
    seed: int = 42,
    audit_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose an offer-disjoint calibration/audit split with two classes each.

    Refuse to quietly fall back to a row-random split if pairs form a single
    large connected component; supply a separate curated dataset instead.
    """
    if not 0 < audit_fraction < 1:
        raise ValueError("audit_fraction must be between 0 and 1")
    labels = pairs["label"].to_numpy(dtype=int)
    if len(pairs) < 8 or not np.isin(labels, [0, 1]).all():
        raise ValueError("need at least 8 labeled binary pairs")
    groups = offer_component_groups(pairs)
    if len(np.unique(groups)) < 4:
        raise ValueError(
            "too few independent offer components to split; collect separate labels"
        )
    for attempt in range(80):
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=audit_fraction, random_state=seed + attempt
        )
        calibration_idx, audit_idx = next(splitter.split(pairs, labels, groups))
        if (
            len(calibration_idx) >= 4
            and len(audit_idx) >= 4
            and len(np.unique(labels[calibration_idx])) == 2
            and len(np.unique(labels[audit_idx])) == 2
        ):
            return np.asarray(calibration_idx), np.asarray(audit_idx)
    raise ValueError(
        "cannot produce two-class offer-disjoint calibration/audit partitions; "
        "use independently labeled validation data"
    )


def _probability_vector(probabilities: object) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("scores must be a nonempty finite probability vector in [0, 1]")
    return values


def probability_logits(probabilities: object) -> np.ndarray:
    """Finite 1-column log-odds for the fitted sigmoid mapping."""
    values = _probability_vector(probabilities)
    safe = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(safe / (1 - safe)).reshape(-1, 1)


def apply_sigmoid_calibration(estimator: object, probabilities: object) -> np.ndarray:
    result = np.asarray(estimator.predict_proba(probability_logits(probabilities))[:, 1])
    return _probability_vector(result)


def reliability_metrics(
    labels: object, probabilities: object, *, n_bins: int = 10
) -> dict[str, float]:
    """Brier score and expected calibration error on a labeled split."""
    y = np.asarray(labels, dtype=int)
    p = _probability_vector(probabilities)
    if y.ndim != 1 or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must align with probabilities and be binary")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    brier = float(np.mean((p - y) ** 2))
    ece = 0.0
    for index in range(n_bins):
        low, high = index / n_bins, (index + 1) / n_bins
        mask = (p >= low) & (p < high if index < n_bins - 1 else p <= high)
        if mask.any():
            ece += float(mask.mean() * abs(y[mask].mean() - p[mask].mean()))
    return {"brier_score": round(brier, 6), "ece": round(ece, 6)}

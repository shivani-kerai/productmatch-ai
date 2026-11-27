"""Session 6 calibration and metrics tests; synthetic fixtures only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from productmatch.api.metrics import RequestMetrics
from productmatch.api.runtime import ServiceConfig, load_calibration
from productmatch.api.service import create_app
from productmatch.evaluation.calibration import (
    apply_sigmoid_calibration,
    file_sha256,
    offer_component_groups,
    probability_logits,
    reliability_metrics,
    split_for_calibration,
)
from productmatch.features.pairwise import FEATURE_NAMES


def make_pairs(n: int = 40) -> pd.DataFrame:
    """Independent two-offer components; balanced classes for calibration."""
    return pd.DataFrame(
        {
            "id_left": np.arange(100, 100 + n * 2, 2),
            "id_right": np.arange(101, 101 + n * 2, 2),
            "label": [i % 2 for i in range(n)],
            "title_left": [f"Product {i}" for i in range(n)],
            "title_right": [f"Product {i}" for i in range(n)],
        }
    )


def test_component_split_is_offer_disjoint() -> None:
    frame = make_pairs()
    calibration, audit = split_for_calibration(frame, seed=42)
    left_offers = set(frame.iloc[calibration]["id_left"]) | set(
        frame.iloc[calibration]["id_right"]
    )
    right_offers = set(frame.iloc[audit]["id_left"]) | set(frame.iloc[audit]["id_right"])
    assert not left_offers & right_offers
    assert len(calibration) + len(audit) == len(frame)
    assert np.array_equal(calibration, split_for_calibration(frame, seed=42)[0])


def test_connected_offers_cannot_cross_groups() -> None:
    frame = pd.DataFrame(
        {
            "id_left": [1, 2, 7],
            "id_right": [2, 3, 8],
        }
    )
    groups = offer_component_groups(frame)
    assert groups[0] == groups[1] != groups[2]


def test_single_component_is_rejected() -> None:
    frame = make_pairs(12)
    frame["id_left"] = 99
    with pytest.raises(ValueError, match="too few independent"):
        split_for_calibration(frame)


def test_reliability_metrics_known_values() -> None:
    metrics = reliability_metrics([0, 1], [0.0, 1.0])
    assert metrics["brier_score"] == 0
    assert metrics["ece"] == 0
    bad = np.array([0.3, np.nan])
    with pytest.raises(ValueError, match="probability"):
        reliability_metrics([0, 1], bad)


def test_log_odds_and_sigmoid() -> None:
    raw = np.array([0.1, 0.2, 0.8, 0.9])
    logits = probability_logits(raw)
    assert logits.shape == (4, 1)
    calibrator = LogisticRegression(random_state=42).fit(
        logits, np.array([0, 0, 1, 1])
    )
    calibrated = apply_sigmoid_calibration(calibrator, raw)
    assert np.isfinite(calibrated).all()
    assert calibrated[0] < calibrated[-1]


def test_model_checksum_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    path.write_bytes(b"model-1")
    assert file_sha256(path) == hashlib.sha256(b"model-1").hexdigest()
    path.write_bytes(b"model-2")
    assert file_sha256(path) != hashlib.sha256(b"model-1").hexdigest()


def test_metrics_labels_are_bounded_and_private() -> None:
    metrics = RequestMetrics()
    metrics.observe("GET", "/health", 200, 0.02)
    metrics.observe("POST", "/private/products/acme-secret", 429, 0.12)
    output = metrics.exposition()
    assert 'route="/health"' in output
    assert 'route="other"' in output
    assert "acme-secret" not in output
    assert 'status="4xx"' in output
    assert 'le="+Inf"' in output
    assert "productmatch_http_duration_seconds_count" in output


@pytest.fixture
def secured_service(tmp_path: Path) -> tuple[TestClient, Path, Path]:
    model_path = tmp_path / "matching.joblib"
    x = pd.DataFrame(
        np.tile(np.arange(2, dtype=float)[:, None], (1, len(FEATURE_NAMES))),
        columns=FEATURE_NAMES,
    )
    model = DummyClassifier(strategy="prior").fit(x, [0, 1])
    joblib.dump(
        {
            "feature_names": FEATURE_NAMES,
            "model": model,
            "model_name": "dummy",
            "decision_threshold": 0.4,
        },
        model_path,
    )

    calibrator = LogisticRegression().fit(np.array([[-3], [-2], [2], [3]]), [0, 0, 1, 1])
    calibration_path = tmp_path / "calibration.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "kind": "sigmoid_logit",
            "model_sha256": file_sha256(model_path),
            "feature_names": tuple(FEATURE_NAMES),
            "decision_threshold": 0.6,
            "estimator": calibrator,
        },
        calibration_path,
    )
    app = create_app(
        ServiceConfig(
            model_path=model_path,
            calibration_path=calibration_path,
            review_db=tmp_path / "feedback.sqlite3",
            api_key="secret-api",
            review_token="secret-review",
        )
    )
    with TestClient(app) as client:
        yield client, model_path, calibration_path


def test_authenticated_metrics_and_model_metadata(secured_service) -> None:
    client, _, _ = secured_service
    assert client.get("/metrics").status_code == 401
    result = client.get("/ready", headers={"X-API-Key": "secret-api"})
    assert result.json()["calibrated"] is True
    assert len(result.json()["model_fingerprint"]) == 12
    result = client.post(
        "/v1/match",
        json={"left": {"title": "Acme 250ml"}, "right": {"title": "Acme 250ml"}},
        headers={"X-API-Key": "secret-api"},
    )
    assert result.status_code == 200
    assert result.json()["calibrated"] is True
    assert result.json()["threshold"] == 0.6
    report = client.get("/metrics", headers={"X-API-Key": "secret-api"})
    assert report.status_code == 200
    assert "productmatch_http_requests_total" in report.text
    assert "Acme" not in report.text


def test_calibration_rejects_wrong_model_sha(secured_service) -> None:
    _, model_path, calibration_path = secured_service
    model_path.write_bytes(b"tampered-model")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_calibration(calibration_path, model_path)


def test_calibration_requires_model(tmp_path: Path) -> None:
    app = create_app(
        ServiceConfig(
            calibration_path=tmp_path / "calibration.joblib",
            review_db=tmp_path / "reviews.sqlite3",
            api_key="secret-api",
            review_token="secret-review",
        )
    )
    with pytest.raises(ValueError, match="calibration requires"):
        with TestClient(app):
            pass


def test_service_refuses_missing_auth_secrets(tmp_path: Path) -> None:
    app = create_app(ServiceConfig(review_db=tmp_path / "reviews.sqlite3"))
    with pytest.raises(RuntimeError, match="API key"):
        with TestClient(app):
            pass

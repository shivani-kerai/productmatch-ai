"""Session 5 API and review workflow tests; no real WDC files or downloads."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from productmatch.api.runtime import ServiceConfig
from productmatch.api.service import create_app
from productmatch.features.pairwise import FEATURE_NAMES

KEY = "test-api-key-only"
REVIEW = "test-review-secret-only"


@pytest.fixture
def catalog_csv(tmp_path: Path) -> Path:
    file = tmp_path / "catalog.csv"
    pd.DataFrame(
        [
            {"offer_id": "a", "title": "Acme Shampoo 250 ml", "brand": "Acme"},
            {"offer_id": "b", "title": "Acme Shampoo 500 ml", "brand": "Acme"},
            {"offer_id": "c", "title": "Beta Camera X200", "brand": "Beta"},
        ]
    ).to_csv(file, index=False)
    return file


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    estimator = DummyClassifier(strategy="prior")
    estimator.fit(
        pd.DataFrame(
            np.array([[1.0] * len(FEATURE_NAMES), [0.0] * len(FEATURE_NAMES)]),
            columns=FEATURE_NAMES,
        ),
        np.array([0, 1]),
    )
    artifact_path = tmp_path / "trusted-local.joblib"
    joblib.dump(
        {
            "model": estimator,
            "feature_names": FEATURE_NAMES,
            "decision_threshold": 0.55,
            "model_name": "synthetic_dummy_model",
        },
        artifact_path,
    )
    return artifact_path


@pytest.fixture
def client(tmp_path: Path, catalog_csv: Path, artifact: Path):
    service = create_app(
        ServiceConfig(
            model_path=artifact,
            catalog_path=catalog_csv,
            review_db=tmp_path / "reviews.sqlite3",
            api_key=KEY,
            review_token=REVIEW,
        )
    )
    with TestClient(service) as test_client:
        yield test_client


def headers(*, reviewer: bool = False) -> dict[str, str]:
    data = {"X-API-Key": KEY}
    if reviewer:
        data["X-Review-Token"] = REVIEW
    return data


def test_health_and_readiness_are_distinct(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").status_code == 401
    ready = client.get("/ready", headers=headers()).json()
    assert ready["model_ready"] is True
    assert ready["catalog_ready"] is True
    assert ready["catalog_size"] == 3


def test_search_requires_authentication(client: TestClient) -> None:
    payload = {"query": {"title": "Acme Shampoo 250ml"}, "top_k": 2}
    assert client.post("/v1/search", json=payload).status_code == 401
    response = client.post("/v1/search", json=payload, headers=headers())
    assert response.status_code == 200
    result = response.json()
    assert result["method"] == "bm25"
    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["offer_id"] == "a"
    assert result["candidates"][0]["score_type"] == "bm25"


def test_score_model_and_no_model(client: TestClient, tmp_path: Path) -> None:
    data = {
        "left": {"title": "Acme Shampoo 250ml"},
        "right": {"title": "Acme Shampoo 500ml"},
    }
    response = client.post("/v1/match", json=data, headers=headers())
    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["match_probability"] <= 1
    assert result["threshold"] == 0.55
    assert result["model_name"] == "synthetic_dummy_model"
    app = create_app(
        ServiceConfig(api_key=KEY, review_token=REVIEW, review_db=tmp_path / "other.sqlite3")
    )
    with TestClient(app) as empty:
        assert empty.post("/v1/match", json=data, headers=headers()).status_code == 503
        assert empty.post(
            "/v1/search",
            json={"query": data["left"]},
            headers=headers(),
        ).status_code == 503


def test_rerank_requires_model_or_scores_candidates(client: TestClient) -> None:
    response = client.post(
        "/v1/search",
        json={"query": {"title": "Acme Shampoo 250 ml"}, "method": "reranked"},
        headers=headers(),
    )
    assert response.status_code == 200
    assert all(
        candidate["score_type"] == "match_probability"
        for candidate in response.json()["candidates"]
    )


def test_review_write_requires_separate_token(client: TestClient) -> None:
    payload = {
        "query": {"title": "Acme Shampoo 250ml"},
        "candidate": {"title": "Acme Shampoo 500ml"},
        "decision": "nonmatch",
        "reviewer": "reviewer-1",
        "notes": "size mismatch",
    }
    assert client.post("/v1/reviews", json=payload, headers=headers()).status_code == 403
    result = client.post("/v1/reviews", json=payload, headers=headers(reviewer=True))
    assert result.status_code == 201
    saved = result.json()
    assert saved["review_id"]
    assert saved["decision"] == "nonmatch"
    recent = client.get("/v1/reviews", headers=headers(reviewer=True))
    assert recent.status_code == 200
    assert recent.json()[0]["review_id"] == saved["review_id"]


def test_strict_payload_validation_and_limits(client: TestClient) -> None:
    response = client.post(
        "/v1/search",
        json={"query": {"title": "Acme"}, "top_k": 100},
        headers=headers(),
    )
    assert response.status_code == 422
    response = client.post(
        "/v1/search",
        json={"query": {"title": "Acme", "secret_field": "leak"}},
        headers=headers(),
    )
    assert response.status_code == 422
    assert client.get(
        "/v1/reviews?limit=1000", headers=headers(reviewer=True)
    ).status_code == 422


def test_reject_incompatible_model(tmp_path: Path, catalog_csv: Path) -> None:
    bad = tmp_path / "bad_model.joblib"
    joblib.dump({"feature_names": ["wrong"], "model": None}, bad)
    app = create_app(
        ServiceConfig(
            model_path=bad,
            catalog_path=catalog_csv,
            review_db=tmp_path / "data.sqlite3",
            api_key=KEY,
            review_token=REVIEW,
        )
    )
    with pytest.raises(ValueError, match="incompatible"):
        with TestClient(app):
            pass


def test_catalog_rejects_duplicate_offer_ids(tmp_path: Path) -> None:
    from productmatch.api.runtime import load_catalog

    file = tmp_path / "duplicate.csv"
    pd.DataFrame(
        [
            {"offer_id": "x", "title": "One"},
            {"offer_id": "x", "title": "Two"},
        ]
    ).to_csv(file, index=False)
    with pytest.raises(ValueError, match="unique"):
        load_catalog(file)


def test_feedback_persists_after_reopen(tmp_path: Path) -> None:
    from productmatch.api.schemas import ReviewSubmission
    from productmatch.api.store import ReviewStore

    path = tmp_path / "feedback.db"
    record = ReviewSubmission(
        query={"title": "Acme Shampoo"},
        candidate={"title": "Acme Conditioner"},
        decision="nonmatch",
        reviewer="reviewer-1",
    )
    saved = ReviewStore(path).add(record)
    results = ReviewStore(path).list(10)
    assert results[0].review_id == saved.review_id
    assert json.loads(results[0].model_dump_json())["candidate"]["title"] == "Acme Conditioner"

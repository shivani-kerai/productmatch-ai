"""Session 7 fixtures: external holdout, evidence and local smoke boundaries."""

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
from productmatch.evaluation.external import (
    audit_holdout,
    evaluate_external_pairs,
    validate_provenance,
)
from productmatch.evaluation.reporting import (
    compile_evidence,
    write_evidence_report,
)
from productmatch.features.pairwise import FEATURE_NAMES
from scripts.smoke_local_api import validate_local_url


@pytest.fixture
def external_data(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    development = tmp_path / "data"
    development.mkdir()
    train = pd.DataFrame(
        {"id_left": [1, 2], "id_right": [3, 4]}
    )
    validation = pd.DataFrame(
        {"id_left": [5, 6], "id_right": [7, 8]}
    )
    train.to_parquet(development / "train.parquet", index=False)
    validation.to_parquet(development / "validation_safe.parquet", index=False)
    pairs_path = tmp_path / "holdout.csv"
    pd.DataFrame(
        [
            {
                "id_left": "new-1", "id_right": "new-2",
                "title_left": "Acme shampoo 250ml",
                "title_right": "Acme shampoo 250 ml", "label": 1,
            },
            {
                "id_left": "new-3", "id_right": "new-4",
                "title_left": "Acme shampoo 250ml",
                "title_right": "Beta DSLR camera 1000", "label": 0,
            },
            {
                "id_left": "new-5", "id_right": "new-6",
                "title_left": "Beta lens X200",
                "title_right": "Beta lens X200", "label": 1,
            },
            {
                "id_left": "new-7", "id_right": "new-8",
                "title_left": "Garden hose 20m",
                "title_right": "Garden hose 50m", "label": 0,
            },
        ]
    ).to_csv(pairs_path, index=False)
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "label_origin": "synthetic_demo",
                "source": "offline synthetic fixture",
                "annotation_protocol": "hardcoded demonstration labels",
                "labeler_count": 1,
            }
        ),
        encoding="utf-8",
    )
    model = DummyClassifier(strategy="prior")
    model.fit(
        pd.DataFrame(
            np.tile(np.arange(2, dtype=float)[:, None], (1, len(FEATURE_NAMES))),
            columns=FEATURE_NAMES,
        ),
        [0, 1],
    )
    model_path = tmp_path / "trusted.joblib"
    joblib.dump(
        {
            "model": model,
            "model_name": "dummy_synthetic",
            "feature_names": FEATURE_NAMES,
            "decision_threshold": 0.5,
        },
        model_path,
    )
    return development, pairs_path, provenance_path, model_path


def test_synthetic_origin_requires_opt_in() -> None:
    metadata = {
        "label_origin": "synthetic_demo",
        "source": "fixture",
        "annotation_protocol": "fixed labels for tests",
        "labeler_count": 1,
    }
    with pytest.raises(ValueError, match="--allow-demo"):
        validate_provenance(metadata, allow_demo=False)
    assert validate_provenance(metadata, allow_demo=True)["verification_status"] == (
        "operator_asserted_not_independently_verified"
    )


def test_external_evaluation_produces_only_synthetic_report(
    external_data: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    root, pairs, meta, model = external_data
    output = tmp_path / "session7" / "scores.json"
    report = evaluate_external_pairs(
        pairs_csv=pairs, provenance_json=meta, model_path=model,
        development_dir=root, report_path=output, allow_demo=True
    )
    assert report["evaluation_type"] == "synthetic_demo"
    assert report["official_wdc_test_sets_read"] is False
    assert report["audit"]["train_offer_overlap"] == 0
    assert report["audit"]["validation_offer_overlap"] == 0
    assert report["no_fitting_or_threshold_tuning_on_external_pairs"] is True
    assert report["metrics"]["raw"]["classification"]["pairs"] == 4
    assert output.exists()
    evidence = write_evidence_report(
        pairwise_path=output, retrieval_path=None, smoke_path=None,
        report_dir=tmp_path / "portfolio"
    )
    assert evidence["pairwise_status"] == "synthetic_demo_only"
    markdown = (tmp_path / "portfolio" / "portfolio_summary.md").read_text()
    assert "Do not cite these numbers" in markdown


def test_reused_offer_is_rejected(
    external_data: tuple[Path, Path, Path, Path]
) -> None:
    root, pairs, _, _ = external_data
    frame = pd.read_csv(pairs, dtype="string")
    frame.loc[0, "id_left"] = "1"
    with pytest.raises(ValueError, match="reused in train"):
        audit_holdout(frame, root)


def test_reversed_duplicate_is_rejected(
    external_data: tuple[Path, Path, Path, Path]
) -> None:
    root, pairs, _, _ = external_data
    frame = pd.read_csv(pairs, dtype="string")
    frame.loc[1, "id_left"] = "new-2"
    frame.loc[1, "id_right"] = "new-1"
    with pytest.raises(ValueError, match="reversed duplicate"):
        audit_holdout(frame, root)


def test_external_data_requires_both_classes(
    external_data: tuple[Path, Path, Path, Path]
) -> None:
    root, pairs, _, _ = external_data
    frame = pd.read_csv(pairs, dtype="string")
    frame["label"] = "1"
    with pytest.raises(ValueError, match="both 0 and 1"):
        audit_holdout(frame, root)


def test_missing_provenance_is_rejected() -> None:
    with pytest.raises(ValueError, match="annotation_protocol"):
        validate_provenance(
            {
                "label_origin": "human_reviewed",
                "source": "annotations",
                "labeler_count": 2,
            },
            allow_demo=False,
        )


def test_empty_portfolio_report_has_no_invented_scores() -> None:
    evidence, markdown = compile_evidence()
    assert evidence["pairwise_status"] == "not_evaluated"
    assert evidence["retrieval_status"] == "not_evaluated"
    assert evidence["production_ready"] is False
    assert "Official WDC final test metrics: **Not evaluated**" in markdown


def test_incomplete_qrels_not_upgraded(tmp_path: Path) -> None:
    report = {
        "complete_qrels": False,
        "results": {
            "bm25": {
                "qrels_complete": False,
                "known_positive_recovery_at_k": {"1": 0.5},
            }
        },
    }
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    evidence, markdown = compile_evidence(retrieval_path=path)
    assert evidence["retrieval_status"] == "incomplete_qrels_diagnostic"
    assert "NOT Recall@K" in markdown
    report["results"]["bm25"]["recall_at_k"] = {"1": 0.5}
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot report"):
        compile_evidence(retrieval_path=path)


def test_local_smoke_url_guard() -> None:
    assert validate_local_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
    for url in (
        "https://127.0.0.1:8000",
        "http://example.com:8000",
        "http://localhost:8000/redirect",
        "http://127.0.0.1:8000?token=abc",
        "http://127.0.0.1",
    ):
        with pytest.raises(ValueError):
            validate_local_url(url)


def test_smoke_works_only_against_operator_loopback(
    tmp_path: Path
) -> None:
    # In-process API contract verification. The CLI smoke still needs a real
    # local server process and never claims it is publicly deployed.
    app = create_app(
        ServiceConfig(
            api_key="private-test",
            review_token="private-review",
            review_db=tmp_path / "reviews.sqlite3",
        )
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 401
        response = client.get("/ready", headers={"X-API-Key": "private-test"})
        assert response.status_code == 200
        assert response.json()["model_ready"] is False
        assert response.json()["catalog_ready"] is False

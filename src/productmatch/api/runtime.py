"""Configured, local-only model and catalog loading for inference.

Only trusted local joblib files selected by the service operator are loaded.
Do not accept artifact paths from API requests or untrusted downloads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from productmatch.evaluation.calibration import apply_sigmoid_calibration, file_sha256
from productmatch.features.pairwise import FEATURE_NAMES, build_pairwise_features
from productmatch.retrieval.benchmark import Offer
from productmatch.retrieval.bm25 import BM25Retriever


@dataclass(frozen=True)
class ServiceConfig:
    model_path: Path | None = None
    calibration_path: Path | None = None
    catalog_path: Path | None = None
    review_db: Path = Path("runtime/reviews.sqlite3")
    api_key: str = ""
    review_token: str = ""

    @classmethod
    def from_environment(cls) -> ServiceConfig:
        """Only deployment configuration may provide local file paths."""
        model = os.getenv("PRODUCTMATCH_MODEL_PATH", "").strip()
        calibration = os.getenv("PRODUCTMATCH_CALIBRATION_PATH", "").strip()
        catalog = os.getenv("PRODUCTMATCH_CATALOG_CSV", "").strip()
        return cls(
            model_path=Path(model) if model else None,
            calibration_path=Path(calibration) if calibration else None,
            catalog_path=Path(catalog) if catalog else None,
            review_db=Path(os.getenv("PRODUCTMATCH_REVIEW_DB", "runtime/reviews.sqlite3")),
            api_key=os.getenv("PRODUCTMATCH_API_KEY", ""),
            review_token=os.getenv("PRODUCTMATCH_REVIEW_TOKEN", ""),
        )


def load_catalog(path: Path) -> tuple[Offer, ...]:
    """Read a curated CSV and reject missing IDs, titles, or duplicated IDs."""
    frame = pd.read_csv(path, dtype="string").fillna("")
    if not {"offer_id", "title"}.issubset(frame.columns):
        raise ValueError("catalog CSV must include offer_id and title")
    if frame.empty:
        raise ValueError("catalog must contain at least one product")
    if frame["offer_id"].str.strip().eq("").any():
        raise ValueError("catalog has blank offer_id")
    if frame["title"].str.strip().eq("").any():
        raise ValueError("catalog has blank title")
    if frame["offer_id"].duplicated().any():
        raise ValueError("catalog offer_id values must be unique")
    offers = []
    for row in frame.to_dict(orient="records"):
        offers.append(
            Offer(
                offer_id=str(row["offer_id"]).strip(),
                title=str(row["title"]).strip(),
                brand=str(row.get("brand", "")).strip(),
                description=str(row.get("description", "")).strip(),
                price=str(row.get("price", "")).strip(),
                currency=str(row.get("currency", "")).strip(),
            )
        )
    return tuple(offers)


def load_model(path: Path) -> tuple[object, float, str]:
    """Load trusted Session 3 estimator after checking the frozen feature schema."""
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError("model artifact must contain a dictionary")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("model artifact has incompatible feature ordering")
    model = payload.get("model")
    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError("model artifact must contain a fitted probability estimator")
    cutoff = float(payload.get("decision_threshold", 0.5))
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("model decision threshold must lie in [0, 1]")
    return model, cutoff, str(payload.get("model_name", "unknown"))


def pair_row(left: Offer, right: Offer) -> dict[str, str]:
    return {
        "title_left": left.title,
        "title_right": right.title,
        "brand_left": left.brand,
        "brand_right": right.brand,
        "description_left": left.description,
        "description_right": right.description,
        "price_left": left.price,
        "price_right": right.price,
        "priceCurrency_left": left.currency,
        "priceCurrency_right": right.currency,
    }


def match_scores(model: object, pairs: list[tuple[Offer, Offer]]) -> list[float]:
    """Score a candidate list in a single batch; no fitting or label access."""
    if not pairs:
        return []
    features = build_pairwise_features(
        pd.DataFrame([pair_row(left, right) for left, right in pairs])
    )
    scores = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    if len(scores) != len(pairs) or not np.isfinite(scores).all():
        raise ValueError("model returned invalid probability values")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("model probability outside [0, 1]")
    return scores.tolist()


def load_calibration(path: Path, model_path: Path) -> tuple[object, float]:
    """Refuse stale calibration when the serving model bytes differ."""
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported calibration artifact schema")
    if payload.get("kind") != "sigmoid_logit":
        raise ValueError("unsupported calibration type")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("calibration feature schema mismatch")
    if payload.get("model_sha256") != file_sha256(model_path):
        raise ValueError("calibration/model SHA256 mismatch; regenerate calibration")
    estimator = payload.get("estimator")
    if estimator is None or not hasattr(estimator, "predict_proba"):
        raise ValueError("calibration artifact missing sigmoid estimator")
    threshold = float(payload.get("decision_threshold", -1))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("invalid calibrated decision threshold")
    return estimator, threshold


class Runtime:
    """Fully initialized state, created only from local operator configuration."""

    def __init__(self, config: ServiceConfig) -> None:
        self.model: object | None = None
        self.model_name: str | None = None
        self.threshold: float | None = None
        self.calibrator: object | None = None
        self.model_fingerprint: str | None = None
        self.catalog: tuple[Offer, ...] = ()
        self.offers: dict[str, Offer] = {}
        self.retriever: BM25Retriever | None = None

        if config.calibration_path is not None and config.model_path is None:
            raise ValueError("calibration requires PRODUCTMATCH_MODEL_PATH")
        if config.model_path is not None:
            self.model, self.threshold, self.model_name = load_model(config.model_path)
            self.model_fingerprint = file_sha256(config.model_path)[:12]
            if config.calibration_path is not None:
                self.calibrator, self.threshold = load_calibration(
                    config.calibration_path, config.model_path
                )
        if config.catalog_path is not None:
            self.catalog = load_catalog(config.catalog_path)
            self.offers = {offer.offer_id: offer for offer in self.catalog}
            self.retriever = BM25Retriever(self.catalog)


    def score_pairs(self, pairs: list[tuple[Offer, Offer]]) -> list[float]:
        """Apply compatible calibration to trusted model probabilities, if configured."""
        if self.model is None:
            raise ValueError("no matching model configured")
        raw = match_scores(self.model, pairs)
        if not raw or self.calibrator is None:
            return raw
        return apply_sigmoid_calibration(self.calibrator, raw).tolist()

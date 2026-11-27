"""Optional supervised pairwise reranking of retrieved candidates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from productmatch.features.pairwise import FEATURE_NAMES, build_pairwise_features
from productmatch.retrieval.benchmark import Offer


def _row(left: Offer, right: Offer) -> dict[str, str]:
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


class PairwiseReranker:
    """Load a Session 3 model fitted on train only; do not fit on test."""

    def __init__(self, artifact: str, catalog: tuple[Offer, ...]) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("Install .[models] for pairwise reranking") from exc
        payload = joblib.load(artifact)
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("model feature schema is incompatible with Session 4")
        self.model = payload["model"]
        self.catalog = {offer.offer_id: offer for offer in catalog}

    def rerank(
        self,
        query: Offer,
        candidates: Sequence[tuple[str, float]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        selected = []
        for offer_id, _ in candidates:
            if offer_id != query.offer_id and offer_id in self.catalog:
                selected.append((offer_id, self.catalog[offer_id]))
        if not selected:
            return []
        features = build_pairwise_features(
            pd.DataFrame([_row(query, offer) for _, offer in selected])
        )
        scores = np.asarray(self.model.predict_proba(features)[:, 1], dtype=float)
        ranked = [(key, float(score)) for (key, _), score in zip(selected, scores, strict=True)]
        return sorted(ranked, key=lambda item: (-item[1], item[0]))[:top_k]


class RerankedRetriever:
    """Retrieve enough candidates, then apply the supervised reranker."""

    def __init__(self, retriever: object, reranker: PairwiseReranker, depth: int = 50):
        if depth < 1:
            raise ValueError("depth must be positive")
        self.retriever, self.reranker, self.depth = retriever, reranker, depth

    def search(self, query: Offer, top_k: int) -> list[tuple[str, float]]:
        return self.reranker.rerank(
            query, self.retriever.search(query, max(top_k, self.depth)), top_k
        )

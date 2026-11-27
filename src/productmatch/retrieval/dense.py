"""Dense retrieval with optional SentenceTransformers and FAISS.

Encoders are injected for offline tests. No external model is downloaded in CI.
The default semantic model is loaded only when the real evaluation script runs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from productmatch.retrieval.benchmark import Offer

Encoder = Callable[[list[str]], np.ndarray]
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def sentence_transformer_encoder(model_name: str = DEFAULT_MODEL) -> Encoder:
    """Construct a lazy runtime embedding encoder; may download model weights."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "SentenceTransformers is not installed. Install .[retrieval] first."
        ) from exc
    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> np.ndarray:
        return np.asarray(
            model.encode(texts, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )

    return encode


def normalized_matrix(values: np.ndarray) -> np.ndarray:
    """Normalize row vectors and reject NaN, zero embeddings and bad shapes."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("embeddings must be a nonempty finite two-dimensional matrix")
    magnitudes = np.linalg.norm(values, axis=1, keepdims=True)
    if (magnitudes <= 0).any():
        raise ValueError("zero-norm embeddings are not supported")
    return np.ascontiguousarray(values / magnitudes, dtype=np.float32)


class DenseRetriever:
    """Exact cosine top-K, using FAISS IndexFlatIP when explicitly enabled.

    The NumPy fallback is an exact brute-force reference, not a scalable ANN index.
    """

    def __init__(
        self,
        catalog: tuple[Offer, ...],
        encoder: Encoder,
        *,
        use_faiss: bool = False,
    ) -> None:
        if not catalog:
            raise ValueError("catalog must not be empty")
        self.catalog = catalog
        self.encoder = encoder
        self.embeddings = normalized_matrix(encoder([o.text for o in catalog]))
        if len(self.embeddings) != len(catalog):
            raise ValueError("encoder returned a different catalog row count")
        self.index = None
        if use_faiss:
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("FAISS missing: pip install faiss-cpu") from exc
            index = faiss.IndexFlatIP(self.embeddings.shape[1])
            index.add(self.embeddings)
            self.index = index

    def search(self, query: Offer, top_k: int) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        vector = normalized_matrix(self.encoder([query.text]))
        if vector.shape[0] != 1 or vector.shape[1] != self.embeddings.shape[1]:
            raise ValueError("query encoder produced wrong embedding shape")
        # FAISS search can return tied top-K in arbitrary order. Score all
        # indexed candidates to keep deterministic tie-breaking by offer ID.
        if self.index is not None:
            _, neighbors = self.index.search(vector, len(self.catalog))
            indices = [int(i) for i in neighbors[0] if i >= 0]
        else:
            indices = list(range(len(self.catalog)))

        scores = self.embeddings[indices] @ vector[0]
        hits = [
            (self.catalog[idx].offer_id, float(score))
            for idx, score in zip(indices, scores, strict=True)
            if self.catalog[idx].offer_id != query.offer_id
        ]
        hits.sort(key=lambda item: (-item[1], item[0]))
        return hits[:top_k]

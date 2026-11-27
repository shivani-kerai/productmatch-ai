"""Pure-Python BM25 candidate generator with deterministic ranking."""

from __future__ import annotations

import math
from collections import Counter

from productmatch.data.preprocessing import normalize_title
from productmatch.retrieval.benchmark import Offer


def tokenize(text: str) -> list[str]:
    return normalize_title(text).split()


class BM25Retriever:
    """Okapi BM25 (positive Robertson-Sparck Jones IDF); no training labels."""

    def __init__(
        self,
        catalog: tuple[Offer, ...],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not catalog:
            raise ValueError("catalog must not be empty")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 parameters must satisfy k1 > 0, 0 <= b <= 1")
        self.catalog = catalog
        self.k1 = k1
        self.b = b
        self.tokens = [Counter(tokenize(offer.text)) for offer in catalog]
        self.lengths = [sum(freq.values()) for freq in self.tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        self.document_freq: Counter[str] = Counter()
        for frequencies in self.tokens:
            self.document_freq.update(frequencies.keys())
        n = len(catalog)
        self.idf = {
            token: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for token, df in self.document_freq.items()
        }

    def search(self, query: Offer, top_k: int) -> list[tuple[str, float]]:
        """Return offer ID / score pairs, excluding a query's identical offer ID."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        tokens = set(tokenize(query.text))
        hits = []
        for offer, frequencies, length in zip(
            self.catalog, self.tokens, self.lengths, strict=True
        ):
            if offer.offer_id == query.offer_id:
                continue
            score = 0.0
            for token in tokens:
                tf = frequencies.get(token, 0)
                if not tf:
                    continue
                norm = 1 - self.b + self.b * (length / self.avgdl if self.avgdl else 0)
                score += self.idf[token] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
            hits.append((offer.offer_id, float(score)))
        hits.sort(key=lambda item: (-item[1], item[0]))
        return hits[:top_k]

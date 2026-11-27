"""Retrieval benchmark construction with explicit judgment-completeness semantics.

WDC pairwise labels do not exhaustively judge a full query/catalog Cartesian
product. Report *known-positive recovery*, not standard catalog Recall@K.
Curated benchmarks may claim complete qrels only when independently audited.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from productmatch.data.preprocessing import normalize_title


@dataclass(frozen=True)
class Offer:
    offer_id: str
    title: str
    brand: str = ""
    description: str = ""
    price: str = ""
    currency: str = ""

    @property
    def text(self) -> str:
        """Canonical, human-readable retrieval text; no IDs or ground truth."""
        return " ".join(
            part for part in (self.brand, self.title, self.description) if part
        ).strip()


@dataclass(frozen=True)
class RetrievalBenchmark:
    catalog: tuple[Offer, ...]
    queries: tuple[Offer, ...]
    qrels: dict[str, frozenset[str]]
    complete_judgments: bool
    source: str

    def __post_init__(self) -> None:
        catalog_ids = [offer.offer_id for offer in self.catalog]
        query_ids = [offer.offer_id for offer in self.queries]
        if not self.catalog or not self.queries:
            raise ValueError("catalog and query sets must not be empty")
        if len(catalog_ids) != len(set(catalog_ids)):
            raise ValueError("catalog offer IDs must be unique")
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("query offer IDs must be unique")
        if set(self.qrels) != set(query_ids):
            raise ValueError("every query must have a qrels entry")
        catalog_set = set(catalog_ids)
        for query_id, relevant in self.qrels.items():
            if not relevant or not relevant <= catalog_set:
                raise ValueError(f"{query_id}: relevant offers must exist in the catalog")
            if query_id in relevant:
                raise ValueError(f"{query_id}: query cannot be relevant to itself")


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _offer(row: dict[str, object], side: str) -> Offer:
    return Offer(
        offer_id=_clean(row[f"id_{side}"]),
        title=_clean(row[f"title_{side}"]),
        brand=_clean(row.get(f"brand_{side}")),
        description=_clean(row.get(f"description_{side}")),
        price=_clean(row.get(f"price_{side}")),
        currency=_clean(row.get(f"priceCurrency_{side}")),
    )


def _save_unique(store: dict[str, Offer], offer: Offer) -> None:
    if not offer.offer_id or not offer.title:
        raise ValueError("offer ID and title are required")
    prior = store.get(offer.offer_id)
    if prior and normalize_title(prior.title) != normalize_title(offer.title):
        raise ValueError(f"offer {offer.offer_id!r} has inconsistent titles")
    store.setdefault(offer.offer_id, offer)


def benchmark_from_wdc_validation(data_dir: Path) -> RetrievalBenchmark:
    """Build an *incomplete-qrels* diagnostic using only validation_safe.

    Left offers with known positive links become queries; ALL right offers from
    the safe validation pairs form a catalog of positives and distractors.
    This is a reproducible known-link recovery exercise, NOT official Recall@K.
    """
    root = Path(data_dir)
    valid_path = root / "validation_safe.parquet"
    if not valid_path.is_file():
        raise FileNotFoundError(
            f"{valid_path} missing; prepare WDC with validation quarantine first"
        )
    frame = pd.read_parquet(valid_path)
    required = {"id_left", "id_right", "title_left", "title_right", "label"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"validation_safe lacks required fields: {sorted(missing)}")
    if frame.empty or not frame["label"].isin([0, 1]).all():
        raise ValueError("validation_safe contains invalid labels")
    catalog: dict[str, Offer] = {}
    queries: dict[str, Offer] = {}
    qrels: dict[str, set[str]] = {}
    for row in frame.to_dict(orient="records"):
        right = _offer(row, "right")
        _save_unique(catalog, right)
        if int(row["label"]) != 1:
            continue
        left = _offer(row, "left")
        _save_unique(queries, left)
        qrels.setdefault(left.offer_id, set()).add(right.offer_id)
    if not qrels:
        raise ValueError("no positive validation pairs: cannot build known-link benchmark")
    return RetrievalBenchmark(
        catalog=tuple(catalog[key] for key in sorted(catalog)),
        queries=tuple(queries[key] for key in sorted(queries)),
        qrels={key: frozenset(values) for key, values in sorted(qrels.items())},
        complete_judgments=False,
        source="WDC validation_safe pairwise positives; incomplete judgments",
    )


def _curated_offers(path: Path) -> tuple[Offer, ...]:
    frame = pd.read_csv(path, dtype="string").fillna("")
    required = {"offer_id", "title"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    selected = []
    for row in frame.to_dict(orient="records"):
        selected.append(
            Offer(
                offer_id=_clean(row["offer_id"]),
                title=_clean(row["title"]),
                brand=_clean(row.get("brand")),
                description=_clean(row.get("description")),
                price=_clean(row.get("price")),
                currency=_clean(row.get("currency")),
            )
        )
    return tuple(selected)


def benchmark_from_curated(
    catalog_csv: Path,
    queries_csv: Path,
    qrels_csv: Path,
    *,
    complete_judgments: bool = False,
) -> RetrievalBenchmark:
    """Load independent query/catalog annotations; require explicit completeness."""
    catalog = _curated_offers(Path(catalog_csv))
    queries = _curated_offers(Path(queries_csv))
    frame = pd.read_csv(qrels_csv, dtype="string").fillna("")
    if not {"query_id", "offer_id"}.issubset(frame):
        raise ValueError("qrels CSV requires query_id and offer_id")
    qrels: dict[str, set[str]] = {}
    for row in frame.to_dict(orient="records"):
        query_id = _clean(row["query_id"])
        offer_id = _clean(row["offer_id"])
        if not query_id or not offer_id:
            raise ValueError("qrels IDs cannot be blank")
        qrels.setdefault(query_id, set()).add(offer_id)
    return RetrievalBenchmark(
        catalog=catalog,
        queries=queries,
        qrels={k: frozenset(v) for k, v in qrels.items()},
        complete_judgments=complete_judgments,
        source="independent curated retrieval labels",
    )

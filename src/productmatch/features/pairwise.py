"""Deterministic, inspectable pairwise product matching features.

Only retail attributes are read. IDs, cluster IDs, pair IDs, labels, hard-negative
flags and official split labels must never become predictive features.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd

from productmatch.data.preprocessing import normalize_title

FEATURE_NAMES = (
    "title_token_jaccard",
    "title_token_containment",
    "title_char_similarity",
    "title_exact",
    "title_common_token_count",
    "title_token_count_gap",
    "brand_known",
    "brand_match",
    "brand_conflict",
    "description_known",
    "description_token_jaccard",
    "numeric_token_conflict",
    "model_token_conflict",
    "quantity_known",
    "quantity_match",
    "quantity_conflict",
    "pack_count_known",
    "pack_count_match",
    "pack_count_conflict",
    "price_comparable",
    "price_relative_gap",
    "currency_known",
    "currency_conflict",
)
ATTRIBUTE_COLUMNS = (
    "title_left",
    "title_right",
    "brand_left",
    "brand_right",
    "description_left",
    "description_right",
    "price_left",
    "price_right",
    "priceCurrency_left",
    "priceCurrency_right",
)
_QUANTITY = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?)\s*(ml|l|kg|g)\b", re.I)
_PACK = re.compile(r"\b(?:pack\s+of\s+(\d+)|(\d+)\s*(?:pack|pcs|pieces))\b", re.I)
_PRICE = re.compile(r"\d+(?:\.\d+)?")
_MODEL = re.compile(r"(?=.*[a-z])(?=.*\d)[a-z\d-]+", re.I)


def _value(value: object) -> str:
    """Convert nullable WDC attributes to text without stringifying NA."""
    return "" if value is None or pd.isna(value) else str(value).strip()


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _quantity(title: str) -> tuple[str, float] | None:
    """Extract first explicit mass or volume quantity, in base units."""
    normalized = unicodedata.normalize("NFKC", title)
    match = _QUANTITY.search(normalized)
    if not match:
        return None
    amount = float(match.group(1).replace(",", "."))
    unit = match.group(2).lower()
    dimension = "volume" if unit in {"ml", "l"} else "mass"
    factor = 1000 if unit in {"l", "kg"} else 1
    return dimension, amount * factor


def _pack_count(title: str) -> int | None:
    match = _PACK.search(unicodedata.normalize("NFKC", title))
    return int(match.group(1) or match.group(2)) if match else None


def _price(price: str) -> float | None:
    """Use price only as a weak relative signal when currencies match."""
    match = _PRICE.search(price.replace(",", ""))
    if not match:
        return None
    amount = float(match.group())
    return amount if 0 < amount < float("inf") else None


def pair_features(row: dict[str, object]) -> dict[str, float]:
    """Extract only features from documented product attributes.

    Missing information is encoded using known/available flags; a missing
    quantity or currency is never asserted to be a confirmed match.
    """
    title_a = _value(row.get("title_left"))
    title_b = _value(row.get("title_right"))
    norm_a = normalize_title(title_a)
    norm_b = normalize_title(title_b)
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    intersection = tokens_a & tokens_b

    brand_a = normalize_title(_value(row.get("brand_left")))
    brand_b = normalize_title(_value(row.get("brand_right")))
    desc_a = set(normalize_title(_value(row.get("description_left"))).split())
    desc_b = set(normalize_title(_value(row.get("description_right"))).split())

    numeric_a = {t for t in tokens_a if any(ch.isdigit() for ch in t)}
    numeric_b = {t for t in tokens_b if any(ch.isdigit() for ch in t)}
    model_a = {t for t in tokens_a if _MODEL.fullmatch(t)}
    model_b = {t for t in tokens_b if _MODEL.fullmatch(t)}

    quantity_a, quantity_b = _quantity(title_a), _quantity(title_b)
    pack_a, pack_b = _pack_count(title_a), _pack_count(title_b)

    currency_a = _value(row.get("priceCurrency_left")).casefold()
    currency_b = _value(row.get("priceCurrency_right")).casefold()
    price_a, price_b = _price(_value(row.get("price_left"))), _price(_value(row.get("price_right")))
    currencies_known = bool(currency_a and currency_b)
    prices_comparable = (
        currencies_known
        and currency_a == currency_b
        and price_a is not None
        and price_b is not None
    )

    feats = {
        "title_token_jaccard": _jaccard(tokens_a, tokens_b),
        "title_token_containment": (
            len(intersection) / min(len(tokens_a), len(tokens_b)) if tokens_a and tokens_b else 0.0
        ),
        "title_char_similarity": SequenceMatcher(None, norm_a, norm_b, autojunk=False).ratio(),
        "title_exact": float(bool(norm_a) and norm_a == norm_b),
        "title_common_token_count": float(len(intersection)),
        "title_token_count_gap": float(abs(len(tokens_a) - len(tokens_b))),
        "brand_known": float(bool(brand_a and brand_b)),
        "brand_match": float(bool(brand_a and brand_b) and brand_a == brand_b),
        "brand_conflict": float(bool(brand_a and brand_b) and brand_a != brand_b),
        "description_known": float(bool(desc_a and desc_b)),
        "description_token_jaccard": _jaccard(desc_a, desc_b),
        "numeric_token_conflict": float(
            bool(numeric_a and numeric_b and not (numeric_a & numeric_b))
        ),
        "model_token_conflict": float(bool(model_a and model_b and not (model_a & model_b))),
        "quantity_known": float(quantity_a is not None and quantity_b is not None),
        "quantity_match": float(
            quantity_a is not None and quantity_b is not None and quantity_a == quantity_b
        ),
        "quantity_conflict": float(
            quantity_a is not None and quantity_b is not None and quantity_a != quantity_b
        ),
        "pack_count_known": float(pack_a is not None and pack_b is not None),
        "pack_count_match": float(pack_a is not None and pack_b is not None and pack_a == pack_b),
        "pack_count_conflict": float(
            pack_a is not None and pack_b is not None and pack_a != pack_b
        ),
        "price_comparable": float(prices_comparable),
        "price_relative_gap": (
            abs(price_a - price_b) / max(price_a, price_b) if prices_comparable else 0.0
        ),
        "currency_known": float(currencies_known),
        "currency_conflict": float(currencies_known and currency_a != currency_b),
    }
    return {name: float(feats[name]) for name in FEATURE_NAMES}


def build_pairwise_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Build a fixed-order numeric feature matrix; never fit on validation."""
    missing = set(("title_left", "title_right")) - set(pairs.columns)
    if missing:
        raise ValueError(f"missing required title attributes: {sorted(missing)}")
    columns = [name for name in ATTRIBUTE_COLUMNS if name in pairs]
    rows = (pair_features(row) for row in pairs.loc[:, columns].to_dict(orient="records"))
    frame = pd.DataFrame.from_records(rows, columns=list(FEATURE_NAMES))
    return frame.astype("float64")

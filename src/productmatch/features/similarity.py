"""Simple lexical features for the first development sprint."""

from productmatch.data.preprocessing import normalize_title


def lexical_features(title_a: str, title_b: str) -> dict[str, object]:
    """Return interpretable text features, NOT a product-match prediction.

    Two empty titles have a Jaccard score of zero; absence of data is not
    evidence that product listings identify the same sellable item.
    """
    normalized_a = normalize_title(title_a)
    normalized_b = normalize_title(title_b)
    tokens_a = set(normalized_a.split())
    tokens_b = set(normalized_b.split())
    union = tokens_a | tokens_b
    intersection = tokens_a & tokens_b

    return {
        "normalized_title_a": normalized_a,
        "normalized_title_b": normalized_b,
        "exact_normalized_title_match": bool(normalized_a) and normalized_a == normalized_b,
        "token_jaccard": round(len(intersection) / len(union), 6) if union else 0.0,
        "common_tokens": sorted(intersection),
        "left_only_tokens": sorted(tokens_a - tokens_b),
        "right_only_tokens": sorted(tokens_b - tokens_a),
    }

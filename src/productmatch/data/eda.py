"""Exploratory profiling for preprocessed WDC product pairs.

Only development data should be used while developing models. Test splits are
intentionally not loaded by the default analysis script.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

EDA_FIELDS = (
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


def profile_split(df: pd.DataFrame) -> dict[str, object]:
    """Summarize class balance, missingness, lengths, and hard negatives."""
    if df.empty:
        raise ValueError("cannot profile empty dataframe")
    required = {"label", "title_left", "title_right"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns for EDA: {sorted(missing)}")

    def missing_rate(column: str) -> float:
        if column not in df.columns:
            return 1.0
        values = df[column].astype("string")
        return round(float((values.isna() | values.str.strip().eq("")).mean()), 4)

    title_lengths = {
        side: {
            "median_tokens": float(
                df[f"title_{side}"].astype("string").str.split().str.len().median()
            ),
            "mean_tokens": round(
                float(df[f"title_{side}"].astype("string").str.split().str.len().mean()), 2
            ),
        }
        for side in ("left", "right")
    }
    result: dict[str, object] = {
        "rows": int(len(df)),
        "positive": int((df["label"] == 1).sum()),
        "negative": int((df["label"] == 0).sum()),
        "positive_rate": round(float((df["label"] == 1).mean()), 4),
        "attribute_missing_fraction": {col: missing_rate(col) for col in EDA_FIELDS},
        "title_token_lengths": title_lengths,
    }
    if "is_hard_negative" in df:
        hard_negatives = df.loc[df["label"] == 0, "is_hard_negative"]
        result["hard_negatives_labeled_true"] = int(hard_negatives.eq(True).sum())
        result["hard_negative_fraction_among_labeled_negatives"] = (
            round(float(hard_negatives.dropna().astype(float).mean()), 4)
            if hard_negatives.notna().any()
            else None
        )
    return result


def development_summary(splits: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    """Return development-set profiles and avoid opening official test sets."""
    if set(splits) != {"train", "validation"}:
        raise ValueError("EDA expects exactly train and validation data")
    return {name: profile_split(splits[name]) for name in ("train", "validation")}

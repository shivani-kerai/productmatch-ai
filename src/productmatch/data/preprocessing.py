"""Minimal, intentionally conservative product-title preprocessing."""

import re
import unicodedata

# Unit suffixes frequently fused to a numeric amount in retail titles.
_UNIT_SUFFIXES = r"(?:ml|kg|g|l|oz|lb)"


def normalize_title(title: str) -> str:
    """Normalize whitespace, punctuation, casing and compact quantity units.

    This is *not* an entity-matching decision. We deliberately preserve numbers,
    letters, and token differences that may distinguish product variants.

    Args:
        title: Raw product title. Non-string inputs raise TypeError.

    Returns:
        Normalized title, possibly an empty string.
    """
    if not isinstance(title, str):
        raise TypeError("title must be a string")

    text = unicodedata.normalize("NFKC", title).casefold()
    text = re.sub(rf"(?<=\d)(?={_UNIT_SUFFIXES}\b)", " ", text)
    text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()

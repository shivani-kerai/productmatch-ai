"""Tests that protect product-identifying information in preprocessing."""

import pytest

from productmatch.data.preprocessing import normalize_title


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Dove Deep Moisture Body-Wash 250ML  ", "dove deep moisture body wash 250 ml"),
        ("Dove Deep Moisture Body Wash 500 ml", "dove deep moisture body wash 500 ml"),
        ("ABC___  2-Pack", "abc 2 pack"),
        ("Ｃａｍｅｒａ  RX480", "camera rx480"),
        ("  ", ""),
    ],
)
def test_normalize_title(raw: str, expected: str) -> None:
    assert normalize_title(raw) == expected


def test_does_not_discard_distinguishing_numbers() -> None:
    assert normalize_title("Body Wash 250ML") != normalize_title("Body Wash 500ML")
    assert normalize_title("Camera RX480") != normalize_title("Camera RX550")


def test_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="title must be a string"):
        normalize_title(None)  # type: ignore[arg-type]

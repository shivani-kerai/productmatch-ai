"""Tests for honest, deterministic lexical features."""

from productmatch.features.similarity import lexical_features


def test_equivalent_formatting_yields_identical_title() -> None:
    features = lexical_features("Dove Body-Wash 250ML", "dove body wash 250 ml")
    assert features["exact_normalized_title_match"] is True
    assert features["token_jaccard"] == 1.0


def test_size_difference_is_visible_in_features() -> None:
    features = lexical_features("Dove Body Wash 250ml", "Dove Body Wash 500ml")
    assert features["exact_normalized_title_match"] is False
    assert features["left_only_tokens"] == ["250"]
    assert features["right_only_tokens"] == ["500"]
    assert 0 < features["token_jaccard"] < 1


def test_empty_titles_do_not_count_as_match() -> None:
    features = lexical_features("", "  ")
    assert features["exact_normalized_title_match"] is False
    assert features["token_jaccard"] == 0.0


def test_feature_extraction_is_order_symmetric_for_jaccard() -> None:
    a_to_b = lexical_features("Dove Body Wash", "Dove Shampoo")
    b_to_a = lexical_features("Dove Shampoo", "Dove Body Wash")
    assert a_to_b["token_jaccard"] == b_to_a["token_jaccard"]

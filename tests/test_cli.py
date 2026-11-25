"""Test both CLI subcommands without requiring an online service."""

import json

from productmatch.cli import main


def test_normalize_cli(capsys) -> None:
    assert main(["normalize", "Dove Body-Wash 250ML"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["normalized"] == "dove body wash 250 ml"


def test_features_cli(capsys) -> None:
    assert main(["features", "Dove 250ml", "Dove 500 ml"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["exact_normalized_title_match"] is False
    assert "250" in result["left_only_tokens"]

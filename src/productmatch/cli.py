"""Minimal, dependency-free demonstration CLI for Session 1."""

import argparse
import json
from collections.abc import Sequence

from productmatch.data.preprocessing import normalize_title
from productmatch.features.similarity import lexical_features


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="productmatch",
        description="ProductMatch AI starter: normalization and lexical features only.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    normalize = commands.add_parser("normalize", help="Normalize a product title")
    normalize.add_argument("title", help="Product title")

    features = commands.add_parser("features", help="Compare two titles using lexical features")
    features.add_argument("title_a", help="First product title")
    features.add_argument("title_b", help="Second product title")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Session 1 CLI and return its process exit status."""
    args = build_parser().parse_args(argv)

    if args.command == "normalize":
        result = {"input": args.title, "normalized": normalize_title(args.title)}
    else:
        result = lexical_features(args.title_a, args.title_b)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0

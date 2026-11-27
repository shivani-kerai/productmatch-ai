"""Assemble an evidence-only local portfolio report (no fabricated metrics)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from productmatch.evaluation.reporting import write_evidence_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise", type=Path)
    parser.add_argument("--retrieval", type=Path)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/session7"))
    args = parser.parse_args()
    result = write_evidence_report(
        pairwise_path=args.pairwise,
        retrieval_path=args.retrieval,
        smoke_path=args.smoke,
        report_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "pairwise_status": result["pairwise_status"],
                "retrieval_status": result["retrieval_status"],
                "deployment_status": result["deployment_status"],
                "production_ready": result["production_ready"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

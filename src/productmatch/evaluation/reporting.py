"""Portfolio evidence report: distinguish confirmed tests from unmeasured claims.

Files are generated locally. Never fabricate WDC scores, publish raw review
data or promote incomplete WDC pairwise-positive retrieval qrels to Recall@K.
"""

from __future__ import annotations

import json
from pathlib import Path


def _load_optional(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evidence report input must contain a JSON object")
    return value


def _value(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    return "Not measured" if value is None else str(value)


def compile_evidence(
    *,
    pairwise_path: Path | None = None,
    retrieval_path: Path | None = None,
    smoke_path: Path | None = None,
) -> tuple[dict[str, object], str]:
    """Build an evidence record and Markdown summary; no benchmark data is invented."""
    pairwise = _load_optional(pairwise_path)
    retrieval = _load_optional(retrieval_path)
    smoke = _load_optional(smoke_path)

    pair_status = "not_evaluated"
    pair_metrics: dict[str, object] = {}
    if pairwise is not None:
        if pairwise.get("schema_version") != 1 or not pairwise.get(
            "no_fitting_or_threshold_tuning_on_external_pairs"
        ):
            raise ValueError("pairwise report does not declare frozen-holdout evaluation")
        if not pairwise.get("audit") or pairwise["audit"].get("identity_audit") != (
            "passed_offer_id_disjointness_only"
        ):
            raise ValueError("pairwise report is missing a clean identity audit")
        if pairwise.get("evaluation_type") == "synthetic_demo":
            pair_status = "synthetic_demo_only"
        elif pairwise.get("evaluation_type") == "independent_pairwise":
            pair_status = "operator_asserted_external_labels"
        else:
            raise ValueError("unrecognized pairwise evaluation type")
        pair_metrics = pairwise.get("metrics", {})

    retrieval_status = "not_evaluated"
    retrieval_metrics: dict[str, object] = {}
    if retrieval is not None:
        if not isinstance(retrieval.get("results"), dict) or not retrieval["results"]:
            raise ValueError("retrieval report requires method results")
        complete = retrieval.get("complete_qrels") is True
        for method_result in retrieval["results"].values():
            if not isinstance(method_result, dict):
                raise ValueError("retrieval method results must be objects")
            if method_result.get("qrels_complete") is not complete:
                raise ValueError("retrieval qrels completeness is inconsistent")
            if complete and "recall_at_k" not in method_result:
                raise ValueError("complete-qrels result must include recall_at_k")
            if not complete and "recall_at_k" in method_result:
                raise ValueError("incomplete qrels cannot report catalog Recall@K")
        retrieval_status = (
            "operator_asserted_complete_qrels" if complete else "incomplete_qrels_diagnostic"
        )
        retrieval_metrics = retrieval["results"]

    smoke_status = "not_checked"
    if smoke is not None:
        if smoke.get("environment") != "local_loopback":
            raise ValueError("smoke evidence is only supported for local loopback")
        smoke_status = "passed_local_smoke" if smoke.get("passed") is True else "failed_local_smoke"

    evidence = {
        "schema_version": 1,
        "project": "ProductMatch AI",
        "session": 7,
        "pairwise_status": pair_status,
        "retrieval_status": retrieval_status,
        "deployment_status": smoke_status,
        "production_ready": False,
        "official_wdc_test_results_claimed": False,
        "limitations": [
            "Human annotation independence and completeness are operator assertions.",
            "The official WDC seen/unseen pairwise test sets remain sealed.",
            "A local API smoke test is not a public/cloud deployment or load test.",
            "Development-only calibration has no independent audit.",
        ],
        "pairwise_metrics": pair_metrics,
        "retrieval_methods": retrieval_metrics,
    }

    lines = [
        "# ProductMatch AI — Evidence-Based Portfolio Summary",
        "",
        "Generated from available local evidence. Missing metrics are intentionally not filled in.",
        "",
        "## Evidence status",
        "",
        f"- Independent pairwise: **{pair_status.replace('_', ' ')}**",
        f"- Retrieval: **{retrieval_status.replace('_', ' ')}**",
        f"- Deployment: **{smoke_status.replace('_', ' ')}**",
        "- Production ready: **No**",
        "- Official WDC final test metrics: **Not evaluated**",
        "",
        "## Pairwise classification",
        "",
    ]
    if pair_status == "not_evaluated":
        lines.append("No independent labeled pairwise evaluation provided.")
    elif pair_status == "synthetic_demo_only":
        lines.append("Synthetic fixture only. **Do not cite these numbers as performance.**")
    else:
        lines.append(
            "Operator-asserted independent labels with automated offer-ID "
            "disjointness; annotation quality is not independently verified."
        )
    if pair_metrics:
        for name, entry in pair_metrics.items():
            summary = entry.get("classification", {}) if isinstance(entry, dict) else {}
            reliability = entry.get("reliability", {}) if isinstance(entry, dict) else {}
            lines.extend(
                [
                    "",
                    f"### {name.title()} model scores",
                    "",
                    "| Metric | Value |",
                    "|---|---:|",
                    f"| F1 at frozen threshold | {_value(summary, 'f1')} |",
                    f"| Precision | {_value(summary, 'precision')} |",
                    f"| Recall | {_value(summary, 'recall')} |",
                    f"| Average precision (AP) | {_value(summary, 'average_precision')} |",
                    f"| Brier score | {_value(reliability, 'brier_score')} |",
                    f"| ECE | {_value(reliability, 'ece')} |",
                ]
            )
    lines.extend(["", "## Retrieval", ""])
    if not retrieval_metrics:
        lines.append("No curated retrieval evaluation supplied.")
    else:
        metric_key = (
            "recall_at_k"
            if retrieval_status == "operator_asserted_complete_qrels"
            else "known_positive_recovery_at_k"
        )
        heading = (
            "Catalog Recall@K (operator-asserted complete judgments)"
            if metric_key == "recall_at_k"
            else "Known-positive recovery@K (incomplete judgments; NOT Recall@K)"
        )
        lines.append(heading)
        for name, item in retrieval_metrics.items():
            cutoffs = item.get(metric_key, {})
            displayed = ", ".join(f"@{k}: {v}" for k, v in sorted(cutoffs.items()))
            lines.append(f"- {name}: {displayed or 'Not measured'}")
    lines.extend(
        [
            "",
            "## Deployment evidence",
            "",
            "Only a local loopback smoke test may be reported here. "
            "No hosted URL, cloud deployment or production SLA is claimed.",
            "",
            "## Next evidence required",
            "",
            "Independent human-reviewed holdout, audited complete query/catalog "
            "relevance judgments, frozen model selection, load tests and approved "
            "private deployment.",
            "",
        ]
    )
    return evidence, "\n".join(lines)


def write_evidence_report(
    *,
    pairwise_path: Path | None,
    retrieval_path: Path | None,
    smoke_path: Path | None,
    report_dir: Path,
) -> dict[str, object]:
    payload, markdown = compile_evidence(
        pairwise_path=pairwise_path,
        retrieval_path=retrieval_path,
        smoke_path=smoke_path,
    )
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "evidence.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "portfolio_summary.md").write_text(markdown, encoding="utf-8")
    return payload

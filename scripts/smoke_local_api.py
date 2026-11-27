"""Read-only local FastAPI smoke test; no public deployment or data modification.

Uses an already-running loopback server, requests without following redirects,
and reports only endpoint status, timing and availability. Secrets and titles
are neither logged nor written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx


def validate_local_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("smoke test only accepts a plain HTTP localhost/127.0.0.1 base URL")
    if parsed.port is None or not 1 <= parsed.port <= 65535:
        raise ValueError("explicit local port required")
    return url.rstrip("/")


def run_smoke(
    *,
    base_url: str,
    api_key: str,
    output: Path,
    timeout: float = 8.0,
) -> dict[str, object]:
    """Check live local endpoints, return sanitized, reproducible report."""
    base = validate_local_url(base_url)
    if not api_key:
        raise ValueError("PRODUCTMATCH_API_KEY required for protected local endpoints")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    checks: list[dict[str, object]] = []
    ready: dict[str, object] = {}
    with httpx.Client(
        base_url=base, timeout=timeout, follow_redirects=False, trust_env=False
    ) as client:

        def check(
            name: str, method: str, route: str, *, protected: bool,
            payload: dict[str, object] | None = None, expected: int = 200
        ) -> httpx.Response:
            headers = {"X-API-Key": api_key} if protected else {}
            start = time.perf_counter()
            response = client.request(method, route, headers=headers, json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000
            correct = response.status_code == expected
            checks.append(
                {
                    "check": name,
                    "status_code": response.status_code,
                    "expected_status_code": expected,
                    "passed": correct,
                    "elapsed_ms": round(elapsed_ms, 3),
                }
            )
            return response

        check("liveness", "GET", "/health", protected=False)
        missing = check("readiness_unauthenticated", "GET", "/ready", protected=False, expected=401)
        assert missing.status_code != 200, "unauthenticated readiness should not be readable"
        response = check("readiness_authenticated", "GET", "/ready", protected=True)
        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("readiness did not contain a JSON object")
            ready = {
                "catalog_ready": data.get("catalog_ready") is True,
                "model_ready": data.get("model_ready") is True,
                "calibrated": data.get("calibrated") is True,
                "catalog_size": data.get("catalog_size"),
            }
        if ready.get("catalog_ready"):
            check(
                "candidate_search", "POST", "/v1/search", protected=True,
                payload={"query": {"title": "Example Shampoo 250ml"}, "top_k": 2},
            )
        if ready.get("model_ready"):
            check(
                "pairwise_model_scoring", "POST", "/v1/match", protected=True,
                payload={
                    "left": {"title": "Example Shampoo 250 ml"},
                    "right": {"title": "Example Shampoo 500 ml"},
                },
            )
        check("protected_metrics", "GET", "/metrics", protected=True)
        check("reviews_requires_separate_token", "GET", "/v1/reviews", protected=True, expected=403)

    required = {"liveness", "readiness_unauthenticated", "readiness_authenticated",
                "protected_metrics", "reviews_requires_separate_token"}
    observed = {entry["check"] for entry in checks}
    passed = required <= observed and all(entry["passed"] for entry in checks)
    report: dict[str, object] = {
        "schema_version": 1,
        "session": 7,
        "environment": "local_loopback",
        "passed": passed,
        "checks": checks,
        "readiness": ready,
        "no_review_writes": True,
        "no_public_deployment_claim": True,
        "limitations": "Local smoke only, not load test, hosted deployment or security certification",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/session7/local_smoke.json")
    )
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()
    result = run_smoke(
        base_url=args.url,
        api_key=os.getenv("PRODUCTMATCH_API_KEY", ""),
        output=args.output,
        timeout=args.timeout,
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "environment": result["environment"],
                "checks": result["checks"],
                "report": str(args.output),
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

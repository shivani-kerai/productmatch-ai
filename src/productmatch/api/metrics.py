"""Bounded in-memory process metrics without user content or secrets.

Metrics are per process; multi-worker deployments require aggregated telemetry.
No raw paths, product titles, queries, reviewer identifiers or API keys become
metric labels or logs.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from threading import Lock

ALLOWED_ROUTES = frozenset(
    {"/health", "/ready", "/metrics", "/v1/search", "/v1/match", "/v1/reviews"}
)
LATENCY_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


class RequestMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counts: Counter[tuple[str, str, str]] = Counter()
        self._bucket_counts: dict[tuple[str, str], list[int]] = defaultdict(
            lambda: [0] * len(LATENCY_BUCKETS)
        )
        self._duration_sum: dict[tuple[str, str], float] = defaultdict(float)
        self._duration_count: Counter[tuple[str, str]] = Counter()

    def observe(self, method: str, path: str, status_code: int, seconds: float) -> None:
        """Keep route, method and status cardinalities bounded."""
        path = path if path in ALLOWED_ROUTES else "other"
        method = method if method in {"GET", "POST"} else "other"
        group = f"{min(max(status_code // 100, 1), 5)}xx"
        duration = max(0.0, float(seconds))
        with self._lock:
            self._counts[(path, method, group)] += 1
            key = (path, method)
            self._duration_count[key] += 1
            self._duration_sum[key] += duration
            bins = self._bucket_counts[key]
            for index, limit in enumerate(LATENCY_BUCKETS):
                if duration <= limit:
                    bins[index] += 1

    def exposition(self) -> str:
        """Prometheus-compatible cumulative histogram, no sensitive labels."""
        lines = [
            "# HELP productmatch_http_requests_total Requests grouped by route and status class.",
            "# TYPE productmatch_http_requests_total counter",
        ]
        with self._lock:
            for (route, method, status_group), count in sorted(self._counts.items()):
                labels = f'route="{route}",method="{method}",status="{status_group}"'
                lines.append(f"productmatch_http_requests_total{{{labels}}} {count}")
            lines += [
                "# HELP productmatch_http_duration_seconds Request wall time in seconds.",
                "# TYPE productmatch_http_duration_seconds histogram",
            ]
            for (route, method), count in sorted(self._duration_count.items()):
                labels = f'route="{route}",method="{method}"'
                bins = self._bucket_counts[(route, method)]
                for boundary, hits in zip(LATENCY_BUCKETS, bins, strict=True):
                    lines.append(
                        f'productmatch_http_duration_seconds_bucket{{{labels},le="{boundary:g}"}} {hits}'
                    )
                lines.append(
                    f'productmatch_http_duration_seconds_bucket{{{labels},le="+Inf"}} {count}'
                )
                lines.append(
                    f"productmatch_http_duration_seconds_sum{{{labels}}} "
                    f"{self._duration_sum[(route, method)]:.6f}"
                )
                lines.append(f"productmatch_http_duration_seconds_count{{{labels}}} {count}")
        return "\n".join(lines) + "\n"


def monotonic_seconds() -> float:
    return time.perf_counter()

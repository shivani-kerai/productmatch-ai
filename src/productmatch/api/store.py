"""SQLite feedback store; review decisions never mutate model weights."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from productmatch.api.schemas import ReviewRecord, ReviewSubmission


class ReviewStore:
    """Small local review queue with atomic writes and no automatic retraining."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    query_json TEXT NOT NULL,
                    candidate_json TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    reviewer TEXT NOT NULL,
                    notes TEXT NOT NULL
                )
                """
            )

    def add(self, submission: ReviewSubmission) -> ReviewRecord:
        record = ReviewRecord(
            **submission.model_dump(),
            review_id=str(uuid.uuid4()),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        import json

        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO reviews
                (review_id, created_at, query_json, candidate_json, decision, reviewer, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.review_id,
                    record.created_at,
                    json.dumps(record.query.model_dump()),
                    json.dumps(record.candidate.model_dump()),
                    record.decision,
                    record.reviewer,
                    record.notes,
                ),
            )
        return record

    def list(self, limit: int = 100) -> list[ReviewRecord]:
        import json

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM reviews
                ORDER BY created_at DESC, review_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            ReviewRecord(
                review_id=row["review_id"],
                created_at=row["created_at"],
                query=json.loads(row["query_json"]),
                candidate=json.loads(row["candidate_json"]),
                decision=row["decision"],
                reviewer=row["reviewer"],
                notes=row["notes"],
            )
            for row in rows
        ]

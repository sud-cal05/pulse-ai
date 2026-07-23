"""
SQLite persistence layer (DESIGN.md section 5.4).

Two tables:
- feedback: one row per FeedbackRecord, keyed by feedback_id (the content
  hash from Phase 3). Re-ingesting the same text always maps to the same
  row (idempotent by construction).
- analysis_cache: one row per (feedback_id, prompt_version, model_version)
  combination. This composite key is deliberate (DESIGN.md section 9): the
  SAME feedback analyzed under a NEW prompt version or a NEW model is
  treated as a cache miss and re-analyzed, while re-running the identical
  pipeline config against already-seen text is always served from cache.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pulseai.schemas.analysis import PerItemAnalysis
from pulseai.schemas.feedback import FeedbackRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id   TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    customer_id   TEXT,
    raw_text      TEXT NOT NULL,
    cleaned_text  TEXT NOT NULL,
    language      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    flags_json    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_cache (
    feedback_id     TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    analysis_json   TEXT NOT NULL,
    analyzed_at     TEXT NOT NULL,
    PRIMARY KEY (feedback_id, prompt_version, model_version)
);
"""


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA)
    return conn


def upsert_feedback_record(conn: sqlite3.Connection, record: FeedbackRecord) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO feedback
            (feedback_id, source, customer_id, raw_text, cleaned_text,
             language, created_at, flags_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.feedback_id,
            record.source,
            record.customer_id,
            record.raw_text,
            record.cleaned_text,
            record.language,
            record.created_at.isoformat(),
            _dump_flags(record.flags),
        ),
    )
    conn.commit()


def get_cached_analysis(
    conn: sqlite3.Connection,
    feedback_id: str,
    prompt_version: str,
    model_version: str,
) -> PerItemAnalysis | None:
    row = conn.execute(
        """
        SELECT analysis_json FROM analysis_cache
        WHERE feedback_id = ? AND prompt_version = ? AND model_version = ?
        """,
        (feedback_id, prompt_version, model_version),
    ).fetchone()

    if row is None:
        return None
    return PerItemAnalysis.model_validate_json(row[0])


def save_analysis(conn: sqlite3.Connection, analysis: PerItemAnalysis) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO analysis_cache
            (feedback_id, prompt_version, model_version, analysis_json, analyzed_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            analysis.feedback_id,
            analysis.prompt_version,
            analysis.model_version,
            analysis.model_dump_json(),
            analysis.analyzed_at.isoformat(),
        ),
    )
    conn.commit()


def get_all_analyses(
    conn: sqlite3.Connection, prompt_version: str, model_version: str
) -> list[PerItemAnalysis]:
    rows = conn.execute(
        """
        SELECT analysis_json FROM analysis_cache
        WHERE prompt_version = ? AND model_version = ?
        """,
        (prompt_version, model_version),
    ).fetchall()
    return [PerItemAnalysis.model_validate_json(r[0]) for r in rows]


def _dump_flags(flags: list[str]) -> str:
    import json
    return json.dumps(flags)

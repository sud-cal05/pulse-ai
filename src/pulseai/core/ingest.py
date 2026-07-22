"""
Stage 1 — Ingestion (DESIGN.md §6).

Why this stage exists: it's the ONLY place that knows about the messy
source format (CSV columns, encodings, missing fields). Everything past
this point works with one uniform shape (RawFeedbackRow), so validate.py
and clean.py never need to know whether the data came from a CSV export,
an API dump, or manual entry.

Failure case handled here: a malformed row (missing required column)
is skipped with a logged reason rather than crashing the whole ingest
(DESIGN.md §6 stage 1, rubric M5B3).
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("pulseai.ingest")


@dataclass
class RawFeedbackRow:
    """
    Intentionally NOT the FeedbackRecord schema yet — cleaned_text and
    language aren't known until clean.py runs. This is the deliberately
    minimal shape that validate.py and clean.py operate on.
    """

    raw_text: str
    source: str
    customer_id: str | None
    created_at: datetime


def load_feedback_csv(path: str | Path) -> list[RawFeedbackRow]:
    """
    Expected CSV columns: feedback_text, customer_id (optional), created_at
    (ISO format, optional — defaults to now if missing/malformed).

    Rows missing feedback_text entirely are skipped and logged, not raised —
    one bad row in a 500-row export shouldn't kill the whole ingest run.
    """
    path = Path(path)
    rows: list[RawFeedbackRow] = []
    skipped = 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            text = (row.get("feedback_text") or "").strip()
            if not text:
                logger.warning("Row %d skipped: missing feedback_text", i)
                skipped += 1
                continue

            created_at = _parse_created_at(row.get("created_at"), row_num=i)

            rows.append(
                RawFeedbackRow(
                    raw_text=text,
                    source="csv",
                    customer_id=(row.get("customer_id") or "").strip() or None,
                    created_at=created_at,
                )
            )

    logger.info("Ingested %d rows from %s (%d skipped)", len(rows), path, skipped)
    return rows


def _parse_created_at(raw_value: str | None, row_num: int) -> datetime:
    if not raw_value:
        return datetime.now()
    try:
        return datetime.fromisoformat(raw_value.strip())
    except ValueError:
        logger.warning(
            "Row %d: could not parse created_at=%r, defaulting to now()",
            row_num, raw_value,
        )
        return datetime.now()
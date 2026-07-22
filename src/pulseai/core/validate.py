"""
Stage 2 - Validation (DESIGN.md section 6, section 16).

Deterministic, code-only checks BEFORE anything is sent to the LLM. This
is where three of the rubric's named edge cases are actually handled:
empty input, very long input, and duplicate feedback.

Design choice: validation never raises on bad input. Every check returns
a decision (skip / truncate+flag / pass) so one malformed row never stops
a batch - that's the same "graceful degradation" principle as the LLM
retry/fallback logic in llm_client.py, just applied one stage earlier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pulseai.core.ingest import RawFeedbackRow
from pulseai.utils.hashing import content_hash

logger = logging.getLogger("pulseai.validate")

MAX_CHARS = 4000


@dataclass
class ValidationResult:
    row: RawFeedbackRow
    action: str
    reason: str | None = None
    content_id: str | None = None


def validate_row(row: RawFeedbackRow) -> ValidationResult:
    text = row.raw_text.strip()

    if not text:
        return ValidationResult(row=row, action="skip", reason="empty_input")

    content_id = content_hash(text)

    if len(text) > MAX_CHARS:
        return ValidationResult(
            row=row, action="truncate", reason="exceeds_max_chars",
            content_id=content_id,
        )

    return ValidationResult(row=row, action="accept", content_id=content_id)


def deduplicate_batch(
    results: list[ValidationResult],
) -> tuple[list[ValidationResult], int]:
    seen: set[str] = set()
    unique: list[ValidationResult] = []
    duplicates = 0

    for result in results:
        if result.action == "skip":
            unique.append(result)
            continue

        if result.content_id in seen:
            duplicates += 1
            logger.info(
                "Duplicate feedback dropped (content_id=%s)", result.content_id
            )
            continue

        seen.add(result.content_id)
        unique.append(result)

    return unique, duplicates

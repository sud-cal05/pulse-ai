"""
Phase 3 tests (DESIGN.md §16, §21 Phase 3 acceptance criteria):
empty / very-long / duplicate / non-English / emoji-only input, all
handled without crashing and with the expected, documented behavior.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from pulseai.core.clean import clean_text, detect_language, is_emoji_only
from pulseai.core.ingest import RawFeedbackRow
from pulseai.core.validate import MAX_CHARS, deduplicate_batch, validate_row


def _row(text: str, customer_id: str = "cust_1") -> RawFeedbackRow:
    return RawFeedbackRow(
        raw_text=text, source="csv", customer_id=customer_id,
        created_at=datetime(2026, 7, 14, 9, 0),
    )


def test_empty_input_is_skipped():
    result = validate_row(_row("   "))
    assert result.action == "skip"
    assert result.reason == "empty_input"


def test_normal_input_is_accepted():
    result = validate_row(_row("The app crashed twice today."))
    assert result.action == "accept"
    assert result.content_id is not None


def test_very_long_input_is_truncated_and_flagged():
    long_text = "This app is great. " * 500  # well over MAX_CHARS
    assert len(long_text) > MAX_CHARS
    result = validate_row(_row(long_text))
    assert result.action == "truncate"
    assert result.reason == "exceeds_max_chars"


def test_duplicate_detection_keeps_first_occurrence():
    rows = [_row("Same feedback text"), _row("Same feedback text"), _row("Different text")]
    results = [validate_row(r) for r in rows]
    deduped, dup_count = deduplicate_batch(results)
    assert dup_count == 1
    assert len(deduped) == 2


def test_emoji_only_detected():
    assert is_emoji_only("😍😍😍") is True
    assert is_emoji_only("great job 😍") is False


def test_language_detection_non_english():
    lang = detect_language("Me encanta la aplicacion pero el soporte tarda demasiado.")
    assert lang != "en"
    assert lang != "unknown"


def test_language_detection_emoji_only_is_unknown():
    assert detect_language("😍😍😍") == "unknown"


def test_clean_text_strips_email_quoting():
    raw = "On Jul 14, 2026, Support wrote:\n> please try again\nStill broken for me."
    cleaned = clean_text(raw)
    assert "wrote:" not in cleaned
    assert ">" not in cleaned
    assert "Still broken for me." in cleaned


def test_clean_text_preserves_wording_and_punctuation():
    raw = "This is AWFUL!!! Fix it now."
    cleaned = clean_text(raw)
    # Cleaning normalizes whitespace only — casing/punctuation must survive,
    # since they carry sentiment signal (DESIGN.md §6 stage 3).
    assert cleaned == raw


def test_consistency_same_input_same_hash():
    """A cheap but important check: two validate_row calls on identical
    text must produce the identical content_id — this IS the mechanism
    behind idempotent caching and consistency (rubric M5B1/M5S2)."""
    a = validate_row(_row("Checkout is broken on mobile"))
    b = validate_row(_row("Checkout is broken on mobile"))
    assert a.content_id == b.content_id


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
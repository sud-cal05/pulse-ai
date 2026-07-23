"""
Phase 6 tests (DESIGN.md section 21 Phase 6 acceptance criteria):
"re-running never re-analyzes a seen item; results persist."
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from pulseai.schemas.analysis import Category, PerItemAnalysis, Sentiment, Urgency
from pulseai.schemas.feedback import FeedbackRecord
from pulseai.storage.cache import analyze_batch_cached
from pulseai.storage.db import (
    get_cached_analysis,
    get_connection,
    save_analysis,
    upsert_feedback_record,
)


class _FakeConfig:
    model = "fake-model-v1"


class FakeLLMClient:
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.config = _FakeConfig()
        self.call_count = 0

    def complete_messages(self, messages: list[dict]) -> str:
        self.call_count += 1
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


VALID_RESPONSE = json.dumps({
    "primary_category": "performance",
    "secondary_categories": [],
    "category_confidence": 0.9,
    "sentiment": "negative",
    "sentiment_score": -0.7,
    "sentiment_confidence": 0.88,
    "urgency": "high",
    "urgency_score": 0.7,
    "urgency_signals": ["blocks checkout"],
    "item_themes": ["checkout timeout"],
    "key_quote": "The checkout page keeps timing out.",
})


def _record(feedback_id: str = "fb_test001", text: str = "The checkout page keeps timing out.") -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id=feedback_id,
        source="csv",
        customer_id="cust_1",
        raw_text=text,
        cleaned_text=text,
        language="en",
        created_at=datetime(2026, 7, 20, 9, 0),
    )


@pytest.fixture
def conn(tmp_path):
    db_path = tmp_path / "test_pulseai.db"
    return get_connection(db_path)


def test_feedback_record_round_trip(conn):
    record = _record()
    upsert_feedback_record(conn, record)

    row = conn.execute(
        "SELECT feedback_id, cleaned_text, language FROM feedback WHERE feedback_id = ?",
        (record.feedback_id,),
    ).fetchone()

    assert row is not None
    assert row[0] == record.feedback_id
    assert row[1] == record.cleaned_text
    assert row[2] == "en"


def test_upsert_is_idempotent_not_duplicated(conn):
    record = _record()
    upsert_feedback_record(conn, record)
    upsert_feedback_record(conn, record)

    count = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE feedback_id = ?", (record.feedback_id,)
    ).fetchone()[0]
    assert count == 1


def test_analysis_cache_round_trip(conn):
    analysis = PerItemAnalysis(
        feedback_id="fb_test001",
        primary_category=Category.PERFORMANCE,
        category_confidence=0.9,
        sentiment=Sentiment.NEGATIVE,
        sentiment_score=-0.7,
        sentiment_confidence=0.88,
        urgency=Urgency.HIGH,
        urgency_score=0.7,
        key_quote="The checkout page keeps timing out.",
        model_version="fake-model-v1",
        prompt_version="analysis_v1",
        analyzed_at=datetime(2026, 7, 20, 9, 5),
    )
    save_analysis(conn, analysis)

    retrieved = get_cached_analysis(conn, "fb_test001", "analysis_v1", "fake-model-v1")
    assert retrieved is not None
    assert retrieved.primary_category == Category.PERFORMANCE
    assert retrieved.sentiment_score == -0.7


def test_different_prompt_version_is_a_cache_miss(conn):
    analysis = PerItemAnalysis(
        feedback_id="fb_test001",
        primary_category=Category.PERFORMANCE,
        category_confidence=0.9,
        sentiment=Sentiment.NEGATIVE,
        sentiment_score=-0.7,
        sentiment_confidence=0.88,
        urgency=Urgency.HIGH,
        urgency_score=0.7,
        key_quote="x",
        model_version="fake-model-v1",
        prompt_version="analysis_v1",
        analyzed_at=datetime(2026, 7, 20, 9, 5),
    )
    save_analysis(conn, analysis)

    result = get_cached_analysis(conn, "fb_test001", "analysis_v2", "fake-model-v1")
    assert result is None


def test_rerun_never_reanalyzes_seen_items(conn):
    records = [_record()]
    client = FakeLLMClient([VALID_RESPONSE])

    results_1, stats_1 = analyze_batch_cached(records, client, conn)
    assert stats_1.hits == 0
    assert stats_1.misses == 1
    assert client.call_count == 1

    results_2, stats_2 = analyze_batch_cached(records, client, conn)
    assert stats_2.hits == 1
    assert stats_2.misses == 0
    assert client.call_count == 1

    assert results_1[0].primary_category == results_2[0].primary_category
    assert results_1[0].sentiment_score == results_2[0].sentiment_score


def test_new_item_in_second_batch_is_a_miss_others_are_hits(conn):
    client = FakeLLMClient([VALID_RESPONSE, VALID_RESPONSE])

    first_batch = [_record(feedback_id="fb_A", text="Text A")]
    analyze_batch_cached(first_batch, client, conn)

    second_batch = [
        _record(feedback_id="fb_A", text="Text A"),
        _record(feedback_id="fb_B", text="Text B"),
    ]
    _, stats = analyze_batch_cached(second_batch, client, conn)

    assert stats.hits == 1
    assert stats.misses == 1
    assert client.call_count == 2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

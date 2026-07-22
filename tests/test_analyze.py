"""
Phase 5 tests (DESIGN.md section 16, section 21 Phase 5 acceptance criteria).
Uses a FakeLLMClient so these run anywhere with no API key required.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from pulseai.ai.llm_client import LLMCallFailed
from pulseai.core.analyze import analyze_feedback_item
from pulseai.schemas.analysis import Category, Sentiment, Urgency
from pulseai.schemas.feedback import FeedbackRecord


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


def _record(text: str = "The checkout page keeps timing out.") -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id="fb_test001",
        source="csv",
        customer_id="cust_1",
        raw_text=text,
        cleaned_text=text,
        language="en",
        created_at=datetime(2026, 7, 20, 9, 0),
    )


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

LOW_CONFIDENCE_RESPONSE = json.dumps({
    "primary_category": "other",
    "secondary_categories": [],
    "category_confidence": 0.3,
    "sentiment": "neutral",
    "sentiment_score": 0.0,
    "sentiment_confidence": 0.4,
    "urgency": "low",
    "urgency_score": 0.1,
    "urgency_signals": [],
    "item_themes": [],
    "key_quote": "unclear",
})


def test_successful_analysis_returns_valid_result():
    client = FakeLLMClient([VALID_RESPONSE])
    result = analyze_feedback_item(_record(), client)

    assert result.primary_category == Category.PERFORMANCE
    assert result.sentiment == Sentiment.NEGATIVE
    assert result.urgency == Urgency.HIGH
    assert result.needs_human_review is False
    assert result.model_version == "fake-model-v1"
    assert result.feedback_id == "fb_test001"


def test_malformed_json_retries_then_succeeds():
    client = FakeLLMClient(["not valid json {{{", VALID_RESPONSE])
    result = analyze_feedback_item(_record(), client)

    assert client.call_count == 2
    assert result.primary_category == Category.PERFORMANCE
    assert result.needs_human_review is False


def test_repeated_malformed_json_falls_back_safely():
    client = FakeLLMClient(["garbage one", "garbage two"])
    result = analyze_feedback_item(_record(), client)

    assert client.call_count == 2
    assert result.primary_category == Category.OTHER
    assert result.needs_human_review is True
    assert result.category_confidence == 0.0


def test_api_failure_falls_back_without_crashing():
    client = FakeLLMClient([LLMCallFailed("simulated outage"), LLMCallFailed("simulated outage")])
    result = analyze_feedback_item(_record(), client)

    assert result.needs_human_review is True
    assert result.primary_category == Category.OTHER


def test_low_confidence_triggers_review_queue():
    client = FakeLLMClient([LOW_CONFIDENCE_RESPONSE])
    result = analyze_feedback_item(_record(), client, confidence_threshold=0.6)

    assert result.needs_human_review is True


def test_high_confidence_does_not_trigger_review_queue():
    client = FakeLLMClient([VALID_RESPONSE])
    result = analyze_feedback_item(_record(), client, confidence_threshold=0.6)

    assert result.needs_human_review is False


def test_consistency_same_input_same_scripted_output_same_result():
    client_a = FakeLLMClient([VALID_RESPONSE])
    client_b = FakeLLMClient([VALID_RESPONSE])

    record = _record()
    result_a = analyze_feedback_item(record, client_a)
    result_b = analyze_feedback_item(record, client_b)

    assert result_a.primary_category == result_b.primary_category
    assert result_a.sentiment == result_b.sentiment
    assert result_a.urgency == result_b.urgency
    assert result_a.category_confidence == result_b.category_confidence
    assert result_a.needs_human_review == result_b.needs_human_review


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

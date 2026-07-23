"""
Phase 9 tests: the grounding validation is the critical path here - fake
citations must be caught and dropped, real ones must survive.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from pulseai.core.summarize import generate_weekly_summary
from pulseai.schemas.themes import RepresentativeQuote, ThemeCluster, WeeklyAggregate


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


def _theme(theme_id: str, quote_ids: list[str]) -> ThemeCluster:
    return ThemeCluster(
        theme_id=theme_id,
        label="checkout timeout on mobile",
        description="Customers report checkout failing on mobile.",
        member_feedback_ids=quote_ids,
        size=len(quote_ids),
        avg_sentiment_score=-0.7,
        avg_urgency_score=0.8,
        representative_quotes=[
            RepresentativeQuote(feedback_id=qid, text=f"quote text for {qid}")
            for qid in quote_ids
        ],
        cohesion=0.9,
        priority_score=1.0,
        period="2026-W29",
    )


def _aggregate(themes: list[ThemeCluster]) -> WeeklyAggregate:
    return WeeklyAggregate(
        period_start=date(2026, 7, 13),
        period_end=date(2026, 7, 19),
        total_feedback=10,
        category_distribution={},
        sentiment_distribution={},
        urgency_distribution={},
        top_themes=themes,
        review_queue_count=1,
    )


def test_summary_with_valid_citations_keeps_all_insights():
    theme = _theme("theme_000", ["fb_1", "fb_2"])
    aggregate = _aggregate([theme])

    response = json.dumps({
        "headline": "Checkout timeouts are the top issue this week.",
        "key_insights": [{
            "statement": "Multiple customers report checkout failing on mobile.",
            "supporting_feedback_ids": ["fb_1", "fb_2"],
            "theme_id": "theme_000",
            "confidence": 0.9,
            "priority_score": 1.0,
        }],
        "recommended_actions": [{
            "action": "Fix mobile checkout timeout.",
            "rationale": "Blocks revenue.",
            "expected_impact": "Affects 20% of feedback.",
            "linked_theme_id": "theme_000",
            "priority": 1,
        }],
        "watch_items": [],
        "caveats": "Small sample size this week.",
    })
    client = FakeLLMClient([response])

    summary = generate_weekly_summary(aggregate, client)

    assert len(summary.key_insights) == 1
    assert summary.key_insights[0].supporting_feedback_ids == ["fb_1", "fb_2"]
    assert len(summary.recommended_actions) == 1


def test_insight_citing_fake_feedback_id_is_dropped():
    theme = _theme("theme_000", ["fb_1", "fb_2"])
    aggregate = _aggregate([theme])

    response = json.dumps({
        "headline": "Checkout timeouts are the top issue this week.",
        "key_insights": [
            {
                "statement": "Real, grounded insight.",
                "supporting_feedback_ids": ["fb_1"],
                "theme_id": "theme_000",
                "confidence": 0.9,
                "priority_score": 1.0,
            },
            {
                "statement": "Fabricated insight citing an ID that doesn't exist.",
                "supporting_feedback_ids": ["fb_999_MADE_UP"],
                "theme_id": "theme_000",
                "confidence": 0.9,
                "priority_score": 0.8,
            },
        ],
        "recommended_actions": [],
        "watch_items": [],
        "caveats": "none",
    })
    client = FakeLLMClient([response])

    summary = generate_weekly_summary(aggregate, client)

    assert len(summary.key_insights) == 1
    assert summary.key_insights[0].statement == "Real, grounded insight."


def test_invalid_theme_id_is_nulled_but_insight_kept():
    theme = _theme("theme_000", ["fb_1"])
    aggregate = _aggregate([theme])

    response = json.dumps({
        "headline": "Headline.",
        "key_insights": [{
            "statement": "Valid citation, bogus theme reference.",
            "supporting_feedback_ids": ["fb_1"],
            "theme_id": "theme_DOES_NOT_EXIST",
            "confidence": 0.8,
            "priority_score": 0.9,
        }],
        "recommended_actions": [],
        "watch_items": [],
        "caveats": "none",
    })
    client = FakeLLMClient([response])

    summary = generate_weekly_summary(aggregate, client)

    assert len(summary.key_insights) == 1
    assert summary.key_insights[0].theme_id is None


def test_repeated_failure_falls_back_to_honest_empty_summary():
    aggregate = _aggregate([_theme("theme_000", ["fb_1"])])
    client = FakeLLMClient(["not json", "still not json"])

    summary = generate_weekly_summary(aggregate, client)

    assert summary.key_insights == []
    assert summary.recommended_actions == []
    assert "failed" in summary.headline.lower()
    assert "could not be generated" in summary.caveats.lower()


def test_retry_succeeds_on_second_attempt():
    aggregate = _aggregate([_theme("theme_000", ["fb_1"])])
    valid_response = json.dumps({
        "headline": "Headline.",
        "key_insights": [],
        "recommended_actions": [],
        "watch_items": [],
        "caveats": "none",
    })
    client = FakeLLMClient(["garbage", valid_response])

    summary = generate_weekly_summary(aggregate, client)

    assert client.call_count == 2
    assert summary.headline == "Headline."


def test_metadata_is_attached_by_code_not_model():
    aggregate = _aggregate([_theme("theme_000", ["fb_1"])])
    response = json.dumps({
        "headline": "H", "key_insights": [], "recommended_actions": [],
        "watch_items": [], "caveats": "none",
    })
    client = FakeLLMClient([response])

    summary = generate_weekly_summary(aggregate, client)

    assert summary.model_version == "fake-model-v1"
    assert summary.prompt_version == "summary_v1"
    assert summary.generated_at is not None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
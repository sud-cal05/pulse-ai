"""
Phase 8 tests (DESIGN.md section 21 Phase 8 acceptance criteria):
"distributions exact and identical on re-run."
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from pulseai.core.aggregate import build_weekly_aggregate, get_iso_week_bounds
from pulseai.schemas.analysis import Category, PerItemAnalysis, Sentiment, Urgency
from pulseai.schemas.themes import ThemeCluster


def _analysis(
    feedback_id: str,
    category: Category,
    sentiment: Sentiment,
    urgency: Urgency,
    needs_review: bool = False,
) -> PerItemAnalysis:
    return PerItemAnalysis(
        feedback_id=feedback_id,
        primary_category=category,
        category_confidence=0.9,
        sentiment=sentiment,
        sentiment_score=0.0,
        sentiment_confidence=0.9,
        urgency=urgency,
        urgency_score=0.5,
        key_quote="x",
        needs_human_review=needs_review,
        model_version="fake-model-v1",
        prompt_version="analysis_v1",
        analyzed_at=datetime(2026, 7, 20, 9, 0),
    )


def _theme(theme_id: str, priority: float) -> ThemeCluster:
    return ThemeCluster(
        theme_id=theme_id,
        label=f"theme {theme_id}",
        description="desc",
        member_feedback_ids=["fb_1"],
        size=1,
        avg_sentiment_score=0.0,
        avg_urgency_score=0.5,
        cohesion=1.0,
        priority_score=priority,
        period="2026-W29",
    )


def test_category_sentiment_urgency_distributions_are_exact_counts():
    analyses = [
        _analysis("fb_1", Category.PERFORMANCE, Sentiment.NEGATIVE, Urgency.HIGH),
        _analysis("fb_2", Category.PERFORMANCE, Sentiment.NEGATIVE, Urgency.MEDIUM),
        _analysis("fb_3", Category.BILLING, Sentiment.POSITIVE, Urgency.LOW),
    ]
    aggregate = build_weekly_aggregate(
        analyses, [], period_start=date(2026, 7, 13), period_end=date(2026, 7, 19)
    )

    assert aggregate.total_feedback == 3
    assert aggregate.category_distribution[Category.PERFORMANCE] == 2
    assert aggregate.category_distribution[Category.BILLING] == 1
    assert aggregate.sentiment_distribution[Sentiment.NEGATIVE] == 2
    assert aggregate.sentiment_distribution[Sentiment.POSITIVE] == 1
    assert aggregate.urgency_distribution[Urgency.HIGH] == 1
    assert aggregate.urgency_distribution[Urgency.MEDIUM] == 1
    assert aggregate.urgency_distribution[Urgency.LOW] == 1


def test_review_queue_count_matches_flagged_items():
    analyses = [
        _analysis("fb_1", Category.OTHER, Sentiment.NEUTRAL, Urgency.LOW, needs_review=True),
        _analysis("fb_2", Category.PERFORMANCE, Sentiment.NEGATIVE, Urgency.HIGH, needs_review=False),
        _analysis("fb_3", Category.OTHER, Sentiment.NEUTRAL, Urgency.LOW, needs_review=True),
    ]
    aggregate = build_weekly_aggregate(
        analyses, [], period_start=date(2026, 7, 13), period_end=date(2026, 7, 19)
    )
    assert aggregate.review_queue_count == 2


def test_top_themes_respects_top_n_and_preserves_order():
    themes = [_theme("t1", 1.0), _theme("t2", 0.8), _theme("t3", 0.5), _theme("t4", 0.1)]
    aggregate = build_weekly_aggregate(
        [], themes, period_start=date(2026, 7, 13), period_end=date(2026, 7, 19),
        top_n_themes=2,
    )
    assert len(aggregate.top_themes) == 2
    assert [t.theme_id for t in aggregate.top_themes] == ["t1", "t2"]


def test_empty_batch_produces_zeroed_aggregate_without_crashing():
    aggregate = build_weekly_aggregate(
        [], [], period_start=date(2026, 7, 13), period_end=date(2026, 7, 19)
    )
    assert aggregate.total_feedback == 0
    assert aggregate.category_distribution == {}
    assert aggregate.review_queue_count == 0
    assert aggregate.top_themes == []


def test_aggregation_is_deterministic_across_repeated_calls():
    """Rubric M5B1/M5S2: identical input, run twice, must match exactly."""
    analyses = [
        _analysis("fb_1", Category.PERFORMANCE, Sentiment.NEGATIVE, Urgency.HIGH),
        _analysis("fb_2", Category.BILLING, Sentiment.POSITIVE, Urgency.LOW),
    ]
    themes = [_theme("t1", 0.9)]

    result_a = build_weekly_aggregate(
        analyses, themes, period_start=date(2026, 7, 13), period_end=date(2026, 7, 19)
    )
    result_b = build_weekly_aggregate(
        analyses, themes, period_start=date(2026, 7, 13), period_end=date(2026, 7, 19)
    )

    assert result_a.model_dump() == result_b.model_dump()


def test_get_iso_week_bounds_known_date():
    # July 22, 2026 is a Wednesday, ISO week 30.
    monday, sunday, period_str = get_iso_week_bounds(date(2026, 7, 22))
    assert monday == date(2026, 7, 20)
    assert sunday == date(2026, 7, 26)
    assert period_str == "2026-W30"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
"""
Phase 2 acceptance test (DESIGN.md §21, Phase 2):
"Pydantic models validate a hand-written example object."

Run with: PYTHONPATH=src python -m pulseai.schemas.validate_examples

This isn't pytest yet (that's Phase 11) — it's a quick manual smoke test
so you can SEE the schemas work before building any pipeline logic on
top of them. Deliberately hand-writes one full example of every schema
in DESIGN.md §8, so you're reading realistic data, not empty stubs.
"""

from __future__ import annotations

from datetime import date, datetime

from pulseai.schemas.analysis import Category, PerItemAnalysis, Sentiment, Urgency
from pulseai.schemas.feedback import FeedbackRecord
from pulseai.schemas.report import (
    DashboardPayload,
    ExecutiveSummary,
    KeyInsight,
    RecommendedAction,
)
from pulseai.schemas.themes import RepresentativeQuote, ThemeCluster, WeeklyAggregate


def build_example_feedback() -> FeedbackRecord:
    return FeedbackRecord(
        feedback_id="fb_8a1c2e",
        source="csv",
        customer_id="cust_442",
        raw_text="checkout keeps timing out on mobile, so frustrating!!",
        cleaned_text="checkout keeps timing out on mobile, so frustrating",
        language="en",
        created_at=datetime(2026, 7, 15, 14, 30),
    )


def build_example_analysis() -> PerItemAnalysis:
    return PerItemAnalysis(
        feedback_id="fb_8a1c2e",
        primary_category=Category.PERFORMANCE,
        secondary_categories=[Category.UI_UX],
        category_confidence=0.91,
        sentiment=Sentiment.NEGATIVE,
        sentiment_score=-0.7,
        sentiment_confidence=0.88,
        urgency=Urgency.HIGH,
        urgency_score=0.75,
        urgency_signals=["blocks checkout completion", "affects revenue path"],
        item_themes=["checkout timeout", "mobile performance"],
        key_quote="checkout keeps timing out on mobile",
        needs_human_review=False,
        model_version="gpt-5-nano",
        prompt_version="analysis_v1",
        analyzed_at=datetime(2026, 7, 15, 14, 31),
    )


def build_example_theme() -> ThemeCluster:
    return ThemeCluster(
        theme_id="theme_001",
        label="checkout timeout on mobile",
        description="Customers report checkout failing/timing out on mobile devices.",
        member_feedback_ids=["fb_8a1c2e", "fb_9b3d4f"],
        size=2,
        avg_sentiment_score=-0.65,
        avg_urgency_score=0.7,
        representative_quotes=[
            RepresentativeQuote(
                feedback_id="fb_8a1c2e",
                text="checkout keeps timing out on mobile",
            )
        ],
        cohesion=0.86,
        priority_score=0.82,
        period="2026-W29",
    )


def build_example_aggregate(theme: ThemeCluster) -> WeeklyAggregate:
    return WeeklyAggregate(
        period_start=date(2026, 7, 13),
        period_end=date(2026, 7, 19),
        total_feedback=120,
        category_distribution={Category.PERFORMANCE: 18, Category.BILLING: 9},
        sentiment_distribution={Sentiment.NEGATIVE: 40, Sentiment.POSITIVE: 60},
        urgency_distribution={Urgency.HIGH: 12, Urgency.CRITICAL: 2},
        top_themes=[theme],
        review_queue_count=7,
    )


def build_example_summary() -> ExecutiveSummary:
    return ExecutiveSummary(
        headline="Mobile checkout timeouts are this week's top revenue risk.",
        key_insights=[
            KeyInsight(
                statement="Checkout timeouts on mobile affect a growing share of negative feedback.",
                supporting_feedback_ids=["fb_8a1c2e", "fb_9b3d4f"],
                theme_id="theme_001",
                confidence=0.85,
                priority_score=0.82,
            )
        ],
        recommended_actions=[
            RecommendedAction(
                action="Prioritize a fix for mobile checkout timeouts.",
                rationale="Directly blocks revenue and shows high urgency signals.",
                expected_impact="Affects ~15% of this week's negative feedback.",
                linked_theme_id="theme_001",
                priority=1,
            )
        ],
        watch_items=["Slight uptick in documentation complaints"],
        caveats="Based on 120 items this week; low volume in some categories limits confidence.",
        generated_at=datetime(2026, 7, 20, 9, 0),
        model_version="gpt-5-nano",
        prompt_version="summary_v1",
    )


def main() -> None:
    feedback = build_example_feedback()
    analysis = build_example_analysis()
    theme = build_example_theme()
    aggregate = build_example_aggregate(theme)
    summary = build_example_summary()

    payload = DashboardPayload(
        aggregate=aggregate,
        summary=summary,
        items=[analysis],
        themes=[theme],
        generated_at=datetime(2026, 7, 20, 9, 5),
    )

    print("All schemas validated successfully.\n")
    print("FeedbackRecord:", feedback.feedback_id)
    print("PerItemAnalysis:", analysis.primary_category, "/", analysis.sentiment,
          "/", analysis.urgency)
    print("ThemeCluster:", theme.label)
    print("WeeklyAggregate: total_feedback =", aggregate.total_feedback)
    print("ExecutiveSummary headline:", summary.headline)
    print("DashboardPayload assembled OK. items:", len(payload.items),
          "themes:", len(payload.themes))


if __name__ == "__main__":
    main()
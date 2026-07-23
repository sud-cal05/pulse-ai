"""
Stage 7 - Aggregation (DESIGN.md section 6, section 11).

Deliberately the simplest, most boring file in the whole pipeline: pure
counting and slicing, zero LLM calls. This is on purpose (DESIGN.md
section 11: "never let the LLM count"). Distributions and totals MUST be
exact and identical on every re-run against the same data - that's a hard
requirement for rubric M5B1/M5S2, and the only way to guarantee it is to
never let a probabilistic model anywhere near the arithmetic.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

from pulseai.schemas.analysis import Category, PerItemAnalysis, Sentiment, Urgency
from pulseai.schemas.themes import ThemeCluster, WeeklyAggregate

DEFAULT_TOP_THEMES = 5


def get_iso_week_bounds(reference_date: date) -> tuple[date, date, str]:
    """
    Given any date, returns (monday_of_that_week, sunday_of_that_week,
    "YYYY-Www"). Used to derive consistent period boundaries for both
    theme clustering (Phase 7's `period` field) and this stage's
    WeeklyAggregate, so the two always agree on which week they're
    describing.
    """
    iso_year, iso_week, _ = reference_date.isocalendar()
    monday = reference_date - timedelta(days=reference_date.isoweekday() - 1)
    sunday = monday + timedelta(days=6)
    period_str = f"{iso_year}-W{iso_week:02d}"
    return monday, sunday, period_str


def build_weekly_aggregate(
    analyses: list[PerItemAnalysis],
    themes: list[ThemeCluster],
    period_start: date,
    period_end: date,
    top_n_themes: int = DEFAULT_TOP_THEMES,
) -> WeeklyAggregate:
    """
    Rolls up a batch of per-item analyses and their theme clusters into
    one WeeklyAggregate. Every number here is a plain count or a slice -
    no model inference, no randomness, no floating hyperparameters.

    `themes` is expected to already be sorted by priority_score descending
    (which build_theme_clusters in Phase 7 guarantees) - this function
    just takes the top N, it doesn't re-sort.
    """
    category_distribution: dict[Category, int] = dict(
        Counter(a.primary_category for a in analyses)
    )
    sentiment_distribution: dict[Sentiment, int] = dict(
        Counter(a.sentiment for a in analyses)
    )
    urgency_distribution: dict[Urgency, int] = dict(
        Counter(a.urgency for a in analyses)
    )

    review_queue_count = sum(1 for a in analyses if a.needs_human_review)

    return WeeklyAggregate(
        period_start=period_start,
        period_end=period_end,
        total_feedback=len(analyses),
        category_distribution=category_distribution,
        sentiment_distribution=sentiment_distribution,
        urgency_distribution=urgency_distribution,
        top_themes=themes[:top_n_themes],
        review_queue_count=review_queue_count,
    )
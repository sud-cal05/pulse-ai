"""
Corpus-level schemas: ThemeCluster (from embeddings clustering, Phase 7)
and WeeklyAggregate (from deterministic code aggregation, Phase 8).

Why WeeklyAggregate is deliberately NOT LLM-generated (DESIGN.md §6 stage 7,
§11): counts and distributions must be exact and identical on re-run. An
LLM asked to "count how many items are negative" is unreliable and
non-deterministic. This object is built entirely with Python/SQL grouping.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from pulseai.schemas.analysis import Category, Sentiment, Urgency


class RepresentativeQuote(BaseModel):
    feedback_id: str
    text: str


class ThemeCluster(BaseModel):
    theme_id: str
    label: str = Field(
        ...,
        description=(
            "MUST be specific ('checkout timeout on mobile'), never generic "
            "('customer issues') — this is graded directly (rubric M5S4)."
        ),
    )
    description: str = Field(..., description="One sentence, plain English.")
    member_feedback_ids: list[str] = Field(default_factory=list)
    size: int = Field(..., ge=0, description="Volume — feeds priority_score.")
    avg_sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    avg_urgency_score: float = Field(..., ge=0.0, le=1.0)
    representative_quotes: list[RepresentativeQuote] = Field(
        default_factory=list,
        description="Grounding material for the weekly executive brief.",
    )
    cohesion: float = Field(
        ..., ge=0.0, le=1.0,
        description="Average intra-cluster similarity — a theme confidence "
        "signal; low cohesion suggests the cluster is a grab-bag, not a "
        "real theme (DESIGN.md §17).",
    )
    priority_score: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "volume_share x avg_urgency_weight x negativity_weight, "
            "normalised. Transparent and explainable, deliberately not a "
            "black-box score (DESIGN.md §3)."
        ),
    )
    period: str = Field(..., description="Week bucket, e.g. '2026-W29'.")


class WeeklyAggregate(BaseModel):
    period_start: date
    period_end: date
    total_feedback: int = Field(..., ge=0)
    category_distribution: dict[Category, int]
    sentiment_distribution: dict[Sentiment, int]
    urgency_distribution: dict[Urgency, int]
    top_themes: list[ThemeCluster] = Field(
        default_factory=list, description="Ranked by priority_score, descending."
    )
    review_queue_count: int = Field(
        ..., ge=0,
        description="How many items the confidence gate flagged for human "
        "review — the honest, visible answer to 'how much can I trust this?'",
    )
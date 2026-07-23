"""
ExecutiveSummary is the hero feature's contract: the Grounded Action Brief.
Every insight MUST cite real feedback_ids. That constraint is what makes
this an anti-hallucination mechanism rather than free-form LLM prose
(DESIGN.md §3, §9). summarize.py (Phase 9) validates every cited ID exists
in the corpus and drops any insight that cites one that doesn't.

DashboardPayload is the single denormalised object the Streamlit UI reads —
the UI stays "dumb" and never touches raw pipeline internals (DESIGN.md §4).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from pulseai.schemas.analysis import PerItemAnalysis
from pulseai.schemas.themes import ThemeCluster, WeeklyAggregate


class KeyInsight(BaseModel):
    statement: str
    supporting_feedback_ids: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "THE grounding mechanism. Every claim must point at real IDs. "
            "summarize.py validates these against the corpus post-generation "
            "and drops any insight citing a nonexistent ID — this is what "
            "lets you tell the mentor 'it structurally cannot hallucinate "
            "a trend' (DESIGN.md §3, §19)."
        ),
    )
    theme_id: str | None = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    priority_score: float = Field(..., ge=0.0, le=1.0)


class RecommendedAction(BaseModel):
    action: str
    rationale: str
    expected_impact: str = Field(
        ..., description="e.g. 'affects ~18% of negative feedback this week'"
    )
    linked_theme_id: str | None = None
    priority: int = Field(..., ge=1, description="1 = highest priority.")

class SummaryLLMOutput(BaseModel):
    """
    The subset of ExecutiveSummary the MODEL produces. generated_at,
    model_version, and prompt_version are attached by code afterward -
    same pattern as LLMAnalysisOutput and ThemeLabelOutput.

    This is also the object that gets GROUNDING-VALIDATED before it's
    allowed to become a real ExecutiveSummary: summarize.py checks every
    supporting_feedback_ids entry against the feedback_ids actually shown
    to the model, and drops (never invents a fix for) any insight that
    cites something it wasn't given.
    """

    headline: str
    key_insights: list[KeyInsight]
    recommended_actions: list[RecommendedAction]
    watch_items: list[str] = Field(default_factory=list)
    caveats: str

class ExecutiveSummary(BaseModel):
    headline: str
    key_insights: list[KeyInsight]
    recommended_actions: list[RecommendedAction]
    watch_items: list[str] = Field(
        default_factory=list,
        description="Emerging or low-volume-but-rising signals worth watching.",
    )
    caveats: str = Field(
        ...,
        description=(
            "Honesty field: sample size limits, low-confidence areas. "
            "Rubric M5D4 rewards knowing what you don't know — this field "
            "makes that an explicit, visible part of the product itself."
        ),
    )
    generated_at: datetime
    model_version: str
    prompt_version: str


class DashboardPayload(BaseModel):
    """The one object the Streamlit dashboard reads. See DESIGN.md §8.6."""

    aggregate: WeeklyAggregate
    summary: ExecutiveSummary
    items: list[PerItemAnalysis] = Field(
        description="Full per-item results, for the drill-down table and "
        "the review queue view."
    )
    themes: list[ThemeCluster]
    generated_at: datetime
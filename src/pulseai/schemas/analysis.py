"""
The classification taxonomy (DESIGN.md §7) and PerItemAnalysis (§8.2) —
the object the LLM must return, schema-enforced, for every feedback item.

Using an Enum (not a free string) for category/sentiment/urgency is a
deliberate anti-hallucination guardrail: the model literally cannot return
a category outside this list once we wire this schema into structured
output at Phase 5. Any attempt to invent a new label fails validation and
is caught by the retry/fallback logic in llm_client.py.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    """DESIGN.md §7 — each category maps to a real business owner, which
    is what lets the demo answer 'who would act on this?' (rubric M5A5)."""

    BUG = "bug_defect"
    PERFORMANCE = "performance"
    FEATURE_REQUEST = "feature_request"
    UI_UX = "ui_ux"
    AUTHENTICATION = "authentication_access"
    BILLING = "billing_pricing"
    DOCUMENTATION = "documentation"
    CUSTOMER_SUPPORT = "customer_support"
    SECURITY = "security_privacy"
    INTEGRATIONS = "integrations"
    PRAISE = "praise_positive"
    OTHER = "other"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    # MIXED exists so genuinely two-sided feedback ("love the UI, hate
    # billing") isn't forced into a misleading single polarity —
    # DESIGN.md §11 calls this out as a maturity signal.
    MIXED = "mixed"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PerItemAnalysis(BaseModel):
    feedback_id: str = Field(..., description="Links back to FeedbackRecord.")

    primary_category: Category = Field(
        ...,
        description=(
            "Single category, chosen by 'what action would the company "
            "take' (DESIGN.md §7), not just keyword matching. Kept singular "
            "so category-distribution charts stay clean and interpretable "
            "(rubric M5S6)."
        ),
    )
    secondary_categories: list[Category] = Field(
        default_factory=list,
        description=(
            "Additional categories for multi-issue feedback, without "
            "polluting the primary distribution used for the main chart."
        ),
    )
    category_confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="Feeds the confidence gate (DESIGN.md §6 stage 5).",
    )

    sentiment: Sentiment
    sentiment_score: float = Field(
        ..., ge=-1.0, le=1.0,
        description="Signed magnitude, for trend lines and heat visuals.",
    )
    sentiment_confidence: float = Field(..., ge=0.0, le=1.0)

    urgency: Urgency
    urgency_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fine-grained ranking within an urgency level.",
    )
    urgency_signals: list[str] = Field(
        default_factory=list,
        description=(
            "WHY this urgency level was chosen (e.g. 'mentions data loss'). "
            "This is what makes urgency explainable instead of a black-box "
            "number — directly supports rubric M5A3/M5A4."
        ),
    )

    item_themes: list[str] = Field(
        default_factory=list,
        description="Short, specific per-item tags — raw material that "
        "theme clustering (Phase 7) groups into corpus-level themes.",
    )
    key_quote: str = Field(
        ...,
        description=(
            "The most representative span of this feedback. Used for "
            "grounding the weekly summary and for dashboard quote cards."
        ),
    )

    needs_human_review: bool = Field(
        default=False,
        description="Set by the confidence gate when any confidence field "
        "falls below the empirically-chosen threshold (DESIGN.md §17).",
    )

    model_version: str = Field(
        ..., description="Pinned model name at analysis time — reproducibility."
    )
    prompt_version: str = Field(
        ..., description="Ties this result to an exact prompt version, so "
        "changing prompts invalidates the right cache entries (DESIGN.md §9)."
    )
    analyzed_at: datetime
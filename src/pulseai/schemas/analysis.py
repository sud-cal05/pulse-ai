"""
The classification taxonomy (DESIGN.md section 7) and PerItemAnalysis
(section 8.2) - the object the LLM must return, schema-enforced, for
every feedback item.

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
    MIXED = "mixed"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PerItemAnalysis(BaseModel):
    feedback_id: str = Field(..., description="Links back to FeedbackRecord.")

    primary_category: Category
    secondary_categories: list[Category] = Field(default_factory=list)
    category_confidence: float = Field(..., ge=0.0, le=1.0)

    sentiment: Sentiment
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sentiment_confidence: float = Field(..., ge=0.0, le=1.0)

    urgency: Urgency
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    urgency_signals: list[str] = Field(default_factory=list)

    item_themes: list[str] = Field(default_factory=list)
    key_quote: str = Field(...)

    needs_human_review: bool = Field(default=False)

    model_version: str = Field(...)
    prompt_version: str = Field(...)
    analyzed_at: datetime


class LLMAnalysisOutput(BaseModel):
    """
    The subset of PerItemAnalysis the MODEL itself is responsible for
    producing. feedback_id, model_version, prompt_version, analyzed_at,
    and needs_human_review are all attached by pipeline code afterward -
    never asked of the model, since it has no way to know them.
    """

    primary_category: Category
    secondary_categories: list[Category] = Field(default_factory=list)
    category_confidence: float = Field(..., ge=0.0, le=1.0)

    sentiment: Sentiment
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sentiment_confidence: float = Field(..., ge=0.0, le=1.0)

    urgency: Urgency
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    urgency_signals: list[str] = Field(default_factory=list)

    item_themes: list[str] = Field(default_factory=list)
    key_quote: str

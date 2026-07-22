"""
FeedbackRecord — the shape of one piece of feedback AFTER ingestion.

Why this schema exists (DESIGN.md §8.1): every downstream stage (analysis,
theme clustering, aggregation) consumes this exact shape, so ingestion is
the ONLY place that has to deal with messy source formats (CSV columns,
API payloads, manual entry). Everything past this point is uniform.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class FeedbackRecord(BaseModel):
    feedback_id: str = Field(
        ...,
        description=(
            "Stable ID, derived from a content hash (see utils/hashing.py). "
            "Using a content hash rather than a row number means re-ingesting "
            "the same feedback twice always yields the same ID — this is "
            "what makes caching and idempotency possible (DESIGN.md §6, §9)."
        ),
    )
    source: str = Field(
        ...,
        description=(
            "Where this record came from: 'csv' | 'api' | 'manual'. "
            "Provenance matters for debugging — if analysis quality drops, "
            "you can check whether it correlates with a specific source."
        ),
    )
    customer_id: str | None = Field(
        default=None,
        description=(
            "Lets aggregation spot a 'one loud customer' inflating a theme, "
            "vs. a genuine trend across many customers (DESIGN.md §18)."
        ),
    )
    raw_text: str = Field(
        ...,
        description=(
            "The ORIGINAL text, never mutated. Kept for auditability — "
            "if cleaning ever looks suspect, you can always compare against "
            "the untouched original."
        ),
    )
    cleaned_text: str = Field(
        ...,
        description="What actually gets sent to the model, after clean.py.",
    )
    language: str = Field(
        default="unknown",
        description=(
            "ISO 639-1 code (e.g. 'en'), or 'unknown'. Drives non-English "
            "handling in validation (DESIGN.md §6 stage 2, §16 edge cases)."
        ),
    )
    created_at: datetime = Field(
        ...,
        description=(
            "Used for weekly bucketing in aggregation and for theme "
            "velocity if the Emerging Issue Radar nice-to-have is built."
        ),
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        allowed = {"csv", "api", "manual"}
        if v not in allowed:
            raise ValueError(f"source must be one of {allowed}, got {v!r}")
        return v
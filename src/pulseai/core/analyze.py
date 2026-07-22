"""
Stage 4 - Per-item analysis (LLM) - and Stage 5 - Confidence gate.
(DESIGN.md section 6, section 9, section 18)

Nothing in this file crashes the batch. A malformed response, a
validation failure, or an exhausted API retry all funnel into the same
safe fallback path (rubric M5B3). The only thing that changes is what
gets logged and what needs_human_review ends up as.

Flow per item:
  1. Build the few-shot-boosted messages (prompt_loader.build_analysis_messages)
  2. Call the LLM, parse the JSON, validate against LLMAnalysisOutput
  3. If step 2 fails for any reason, retry ONCE with a fresh call
  4. If that also fails, fall back to a safe, clearly-flagged default
  5. Apply the confidence gate: low confidence -> needs_human_review = True
  6. Merge with pipeline-supplied fields to build the final PerItemAnalysis
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from pydantic import ValidationError

from pulseai.ai.llm_client import LLMCallFailed, LLMClient
from pulseai.ai.prompt_loader import SYSTEM_PROMPT_VERSION, build_analysis_messages
from pulseai.schemas.analysis import (
    Category,
    LLMAnalysisOutput,
    PerItemAnalysis,
    Sentiment,
    Urgency,
)
from pulseai.schemas.feedback import FeedbackRecord

logger = logging.getLogger("pulseai.analyze")

# Below this, an item is routed to human review rather than trusted as-is.
# DESIGN.md section 17: placeholder default - Phase 11's calibration
# analysis should set this threshold from data, not a guess.
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


def _parse_and_validate(raw_response: str) -> LLMAnalysisOutput:
    """Raises (json.JSONDecodeError, ValidationError) on failure - the
    caller decides what to do about it (retry / fallback)."""
    parsed = json.loads(raw_response)
    return LLMAnalysisOutput(**parsed)


def _safe_fallback(record: FeedbackRecord) -> LLMAnalysisOutput:
    """
    Last-resort path when the LLM call and its retry both fail. The item
    still gets a valid, schema-conformant result - it's just an honest
    "I don't know" routed straight to human review rather than a guessed
    label (DESIGN.md section 18).
    """
    logger.warning(
        "Falling back to safe default for feedback_id=%s after repeated "
        "failure to get a valid analysis.", record.feedback_id,
    )
    return LLMAnalysisOutput(
        primary_category=Category.OTHER,
        secondary_categories=[],
        category_confidence=0.0,
        sentiment=Sentiment.NEUTRAL,
        sentiment_score=0.0,
        sentiment_confidence=0.0,
        urgency=Urgency.LOW,
        urgency_score=0.0,
        urgency_signals=["automated analysis failed; needs manual review"],
        item_themes=[],
        key_quote=record.cleaned_text[:200],
    )


def _get_llm_output(record: FeedbackRecord, client: LLMClient) -> LLMAnalysisOutput:
    """
    One attempt, then one retry, then fallback. This is a DIFFERENT retry
    layer from the one inside llm_client.py: that one retries on transient
    network errors for a single call. This one retries the WHOLE
    call-parse-validate sequence, because a successful network call can
    still come back with malformed or schema-invalid JSON.
    """
    messages = build_analysis_messages(record.cleaned_text)

    for attempt in (1, 2):
        try:
            raw = client.complete_messages(messages)
            return _parse_and_validate(raw)
        except LLMCallFailed as exc:
            logger.warning(
                "Attempt %d/2 failed for feedback_id=%s: LLM call failed (%s)",
                attempt, record.feedback_id, exc,
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "Attempt %d/2 failed for feedback_id=%s: invalid response (%s)",
                attempt, record.feedback_id, exc,
            )

    return _safe_fallback(record)


def analyze_feedback_item(
    record: FeedbackRecord,
    client: LLMClient,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> PerItemAnalysis:
    """
    Analyzes a single feedback item end to end and returns a full,
    schema-valid PerItemAnalysis - this NEVER raises.
    """
    llm_output = _get_llm_output(record, client)

    # Confidence gate (DESIGN.md section 6 stage 5): the WEAKEST confidence
    # across category/sentiment governs review-routing.
    weakest_confidence = min(
        llm_output.category_confidence, llm_output.sentiment_confidence
    )
    needs_human_review = weakest_confidence < confidence_threshold

    return PerItemAnalysis(
        feedback_id=record.feedback_id,
        primary_category=llm_output.primary_category,
        secondary_categories=llm_output.secondary_categories,
        category_confidence=llm_output.category_confidence,
        sentiment=llm_output.sentiment,
        sentiment_score=llm_output.sentiment_score,
        sentiment_confidence=llm_output.sentiment_confidence,
        urgency=llm_output.urgency,
        urgency_score=llm_output.urgency_score,
        urgency_signals=llm_output.urgency_signals,
        item_themes=llm_output.item_themes,
        key_quote=llm_output.key_quote,
        needs_human_review=needs_human_review,
        model_version=client.config.model,
        prompt_version=SYSTEM_PROMPT_VERSION,
        analyzed_at=datetime.now(),
    )


def analyze_batch(
    records: list[FeedbackRecord],
    client: LLMClient,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[PerItemAnalysis]:
    """
    Sequential batch analysis. NOTE: this is a deliberate simplification
    for Phase 5 - running items one at a time is easiest to get correct
    first. Dispatching concurrently (10-20 at a time) is a documented
    future optimization once correctness is nailed down.
    """
    results = []
    for record in records:
        results.append(analyze_feedback_item(record, client, confidence_threshold))
    return results

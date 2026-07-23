"""
Stage 8 - Weekly summary: the Grounded Action Brief.

The hero feature earns its name through one mechanism: every insight the
model produces is checked against the set of feedback_ids it was
ACTUALLY SHOWN. Any insight citing an ID it wasn't given gets dropped -
not "fixed", not silently re-attributed, dropped.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from pydantic import ValidationError

from pulseai.ai.llm_client import LLMCallFailed, LLMClient
from pulseai.ai.prompt_loader import SUMMARY_PROMPT_VERSION, build_summary_messages
from pulseai.schemas.report import ExecutiveSummary, SummaryLLMOutput
from pulseai.schemas.themes import WeeklyAggregate

logger = logging.getLogger("pulseai.summarize")


def _fallback_summary_output() -> SummaryLLMOutput:
    logger.warning("Falling back to empty summary after repeated generation failure.")
    return SummaryLLMOutput(
        headline="Automated summary generation failed this run.",
        key_insights=[],
        recommended_actions=[],
        watch_items=[],
        caveats="The executive summary could not be generated automatically "
        "this run. Review the raw category/sentiment/urgency distributions "
        "and top themes directly until this is resolved.",
    )


def _get_summary_llm_output(aggregate: WeeklyAggregate, client: LLMClient) -> SummaryLLMOutput:
    messages = build_summary_messages(aggregate)

    for attempt in (1, 2):
        try:
            raw = client.complete_messages(messages)
            parsed = json.loads(raw)
            return SummaryLLMOutput(**parsed)
        except LLMCallFailed as exc:
            logger.warning("Summary attempt %d/2 failed: LLM call failed (%s)", attempt, exc)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Summary attempt %d/2 failed: invalid response (%s)", attempt, exc)

    return _fallback_summary_output()


def _validate_grounding(
    llm_output: SummaryLLMOutput,
    valid_feedback_ids: set[str],
    valid_theme_ids: set[str],
) -> SummaryLLMOutput:
    kept_insights = []
    for insight in llm_output.key_insights:
        cited_ids = set(insight.supporting_feedback_ids)
        invalid_ids = cited_ids - valid_feedback_ids
        if invalid_ids:
            logger.warning(
                "Dropping ungrounded insight (cited unknown feedback_ids %s): %r",
                invalid_ids, insight.statement,
            )
            continue

        if insight.theme_id is not None and insight.theme_id not in valid_theme_ids:
            logger.warning(
                "Nulling invalid theme_id %r on insight: %r",
                insight.theme_id, insight.statement,
            )
            insight = insight.model_copy(update={"theme_id": None})

        kept_insights.append(insight)

    kept_actions = []
    for action in llm_output.recommended_actions:
        if action.linked_theme_id is not None and action.linked_theme_id not in valid_theme_ids:
            logger.warning(
                "Nulling invalid linked_theme_id %r on action: %r",
                action.linked_theme_id, action.action,
            )
            action = action.model_copy(update={"linked_theme_id": None})
        kept_actions.append(action)

    return llm_output.model_copy(
        update={"key_insights": kept_insights, "recommended_actions": kept_actions}
    )


def generate_weekly_summary(aggregate: WeeklyAggregate, client: LLMClient) -> ExecutiveSummary:
    valid_feedback_ids = {
        quote.feedback_id
        for theme in aggregate.top_themes
        for quote in theme.representative_quotes
    }
    valid_theme_ids = {theme.theme_id for theme in aggregate.top_themes}

    raw_output = _get_summary_llm_output(aggregate, client)
    grounded_output = _validate_grounding(raw_output, valid_feedback_ids, valid_theme_ids)

    return ExecutiveSummary(
        headline=grounded_output.headline,
        key_insights=grounded_output.key_insights,
        recommended_actions=grounded_output.recommended_actions,
        watch_items=grounded_output.watch_items,
        caveats=grounded_output.caveats,
        generated_at=datetime.now(),
        model_version=client.config.model,
        prompt_version=SUMMARY_PROMPT_VERSION,
    )
"""
Assembles the versioned system prompt + few-shot examples into a chat
message list. Kept separate from llm_client.py (which knows nothing
about prompt content).
"""

from __future__ import annotations

import json
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

SYSTEM_PROMPT_VERSION = "analysis_v2"


def load_system_prompt(version: str = SYSTEM_PROMPT_VERSION) -> str:
    path = PROMPTS_DIR / f"system_{version}.txt"
    return path.read_text(encoding="utf-8").strip()


def load_fewshot_examples(version: str = SYSTEM_PROMPT_VERSION) -> list[dict]:
    path = PROMPTS_DIR / f"fewshot_{version}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_analysis_messages(feedback_text: str, version: str = SYSTEM_PROMPT_VERSION) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": load_system_prompt(version)}
    ]

    for example in load_fewshot_examples(version):
        messages.append(
            {"role": "user", "content": f"Feedback: {example['input']}"}
        )
        messages.append(
            {"role": "assistant", "content": json.dumps(example["output"])}
        )

    messages.append({"role": "user", "content": f"Feedback: {feedback_text}"})
    return messages


THEME_LABEL_PROMPT_VERSION = "theme_labeling_v1"


def build_theme_labeling_messages(
    cluster_themes: list[list[str]], cluster_quotes: list[str]
) -> list[dict]:
    """
    Builds the message list for labeling ONE theme cluster. No few-shot
    examples here — labeling from real member data is a more constrained
    task where the main risk is genericness, handled by the system
    prompt's explicit rules instead.
    """
    system = load_system_prompt(version=THEME_LABEL_PROMPT_VERSION)
    lines = []
    for i, (themes, quote) in enumerate(zip(cluster_themes, cluster_quotes), start=1):
        theme_str = ", ".join(themes) if themes else "(no specific themes tagged)"
        lines.append(f"Item {i} themes: {theme_str}")
        lines.append(f'Item {i} quote: "{quote}"')

    user_content = "\n".join(lines)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

SUMMARY_PROMPT_VERSION = "summary_v1"


def _pct(count: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{count / total * 100:.0f}%"


def build_summary_messages(aggregate) -> list[dict]:
    """
    Renders the DETERMINISTIC WeeklyAggregate (already computed in code -
    Phase 8) into a compact factual brief the LLM turns into narrative.
    The LLM never sees raw feedback or does any counting - it only ever
    sees numbers that are already correct, plus a small set of
    representative quotes with their feedback_ids, which become the ONLY
    valid citation targets for grounding (enforced in core/summarize.py).
    """
    system = load_system_prompt(version=SUMMARY_PROMPT_VERSION)
    total = aggregate.total_feedback
    lines = [
        f"Period: {aggregate.period_start} to {aggregate.period_end}",
        f"Total feedback: {total}",
        f"Items flagged for human review (low confidence): {aggregate.review_queue_count}",
        "",
        "Category distribution:",
    ]
    for cat, count in aggregate.category_distribution.items():
        lines.append(f"  - {cat.value}: {count} ({_pct(count, total)})")

    lines.append("")
    lines.append("Sentiment distribution:")
    for sent, count in aggregate.sentiment_distribution.items():
        lines.append(f"  - {sent.value}: {count} ({_pct(count, total)})")

    lines.append("")
    lines.append("Urgency distribution:")
    for urg, count in aggregate.urgency_distribution.items():
        lines.append(f"  - {urg.value}: {count} ({_pct(count, total)})")

    lines.append("")
    lines.append("Top themes (ranked by priority score):")
    for theme in aggregate.top_themes:
        share = _pct(theme.size, total)
        lines.append("")
        lines.append(f'Theme {theme.theme_id}: "{theme.label}"')
        lines.append(f"  Description: {theme.description}")
        lines.append(f"  Size: {theme.size} items ({share} of total volume)")
        lines.append(f"  Avg sentiment score: {theme.avg_sentiment_score:.2f} (-1=very negative, +1=very positive)")
        lines.append(f"  Avg urgency score: {theme.avg_urgency_score:.2f} (0-1 scale)")
        lines.append(f"  Priority score: {theme.priority_score:.2f}")
        lines.append(f"  Cohesion: {theme.cohesion:.2f}")
        for q in theme.representative_quotes:
            lines.append(f'  Quote [{q.feedback_id}]: "{q.text}"')

    user_content = "\n".join(lines)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
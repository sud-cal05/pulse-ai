"""
Assembles the versioned system prompt + few-shot examples into a chat
message list. Kept separate from llm_client.py (which knows nothing
about prompt content).
"""

from __future__ import annotations

import json
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

SYSTEM_PROMPT_VERSION = "analysis_v1"


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

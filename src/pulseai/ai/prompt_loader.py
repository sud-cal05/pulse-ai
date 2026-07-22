"""
Assembles the versioned system prompt + few-shot examples (DESIGN.md §9,
§10) into a chat message list. Kept separate from llm_client.py (which
knows nothing about prompt content) and from analyze.py (Phase 5, which
will call this and then hand the result to the LLM client).

Why few-shot examples are sent as alternating user/assistant turns rather
than pasted into the system prompt as text: this is the standard pattern
for chat-completion APIs, and it lets the model see the exact input->JSON
mapping in the same format it'll be asked to produce for the real input.
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
    """
    Loads the few-shot JSON. Each example carries a "_targets" key purely
    for human documentation (which failure mode it defends against,
    DESIGN.md §10/§18) — that key is NEVER sent to the model; it's
    stripped out in build_analysis_messages below.
    """
    path = PROMPTS_DIR / f"fewshot_{version}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_analysis_messages(feedback_text: str, version: str = SYSTEM_PROMPT_VERSION) -> list[dict]:
    """
    Returns the full message list for one analysis call: system prompt,
    then each few-shot example as a user/assistant turn pair, then the
    real feedback as the final user turn.
    """
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
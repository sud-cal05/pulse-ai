"""
Zero-shot probe (DESIGN.md section 10 process, section 21 Phase 4).

Run this LOCALLY with a real OPENAI_API_KEY (it can't run in a sandboxed
environment without live network access to the provider).

What it does: sends each sample feedback item to the model TWICE -
once with only the system prompt (zero-shot) and once with the full
few-shot-boosted prompt - so you can see, side by side, whether the
few-shot examples actually change the answer on the hard cases
(sarcasm, negation, mixed sentiment, category boundaries).

Usage:
    PYTHONPATH=src python -m pulseai.prompts.zero_shot_probe
"""

from __future__ import annotations

import json

from pulseai.ai.llm_client import LLMCallFailed, LLMClient
from pulseai.ai.prompt_loader import build_analysis_messages, load_system_prompt

PROBE_ITEMS = [
    ("sarcasm", "Oh fantastic, the export feature broke again right before my board meeting. Perfect."),
    ("negation", "Honestly the new pricing page isn't as confusing as I expected, found the plan I needed fast."),
    ("mixed sentiment", "The mobile app looks great now but I can't actually complete a purchase, checkout just spins forever."),
    ("category boundary (perf vs bug)", "Searching for a customer record sometimes just hangs for 30+ seconds before anything shows up."),
    ("category boundary (feature vs bug)", "Would be nice if exporting to CSV actually included all the rows instead of cutting off at 100."),
]


def _zero_shot_messages(feedback_text: str) -> list[dict]:
    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": f"Feedback: {feedback_text}"},
    ]


def run_probe() -> None:
    client = LLMClient()
    print(f"Using model: {client.config.model}\n")

    for label, text in PROBE_ITEMS:
        print("=" * 78)
        print(f"CASE: {label}")
        print(f"INPUT: {text}\n")

        for mode, messages in [
            ("ZERO-SHOT", _zero_shot_messages(text)),
            ("FEW-SHOT", build_analysis_messages(text)),
        ]:
            try:
                raw = client.complete_messages(messages)
                parsed = json.loads(raw)
                print(
                    f"  [{mode}] sentiment={parsed.get('sentiment')} "
                    f"category={parsed.get('primary_category')} "
                    f"urgency={parsed.get('urgency')}"
                )
            except (LLMCallFailed, json.JSONDecodeError) as exc:
                print(f"  [{mode}] FAILED: {exc}")
        print()

    print("=" * 78)
    print(
        "Review above: for each hard case, did FEW-SHOT correct a wrong "
        "ZERO-SHOT reading? If a case still looks wrong even with few-shot, "
        "that's your signal to add a new targeted example in a v2 prompt "
        "rather than editing v1 in place."
    )


if __name__ == "__main__":
    run_probe()

"""
Phase 4 acceptance test (DESIGN.md §21, Phase 4).

Two checks:
1. Every few-shot example's "output" is schema-valid against
   LLMAnalysisOutput. If a hand-written example doesn't validate, the
   model would be trained on a broken pattern — this must never happen.
2. build_analysis_messages assembles a well-formed message list.

Run with: PYTHONPATH=src python -m pulseai.prompts.validate_fewshot
"""

from __future__ import annotations

from pulseai.ai.prompt_loader import build_analysis_messages, load_fewshot_examples
from pulseai.schemas.analysis import LLMAnalysisOutput


def main() -> None:
    examples = load_fewshot_examples()
    print(f"Loaded {len(examples)} few-shot examples.\n")

    for i, example in enumerate(examples, start=1):
        target = example.get("_targets", "(no _targets note)")
        try:
            LLMAnalysisOutput(**example["output"])
            print(f"[{i:2d}] PASS  — targets: {target}")
        except Exception as exc:
            print(f"[{i:2d}] FAIL  — targets: {target}\n       {exc}")
            raise

    print(f"\nAll {len(examples)} few-shot examples are schema-valid.\n")

    messages = build_analysis_messages("This is a placeholder real feedback item.")
    expected_len = 1 + (len(examples) * 2) + 1  # system + pairs + final user turn
    assert len(messages) == expected_len, (
        f"expected {expected_len} messages, got {len(messages)}"
    )
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "_targets" not in messages[1]["content"]  # metadata must not leak to the model
    print(f"Message assembly OK: {len(messages)} messages "
          f"(1 system + {len(examples)} example pairs + 1 real input).")


if __name__ == "__main__":
    main()
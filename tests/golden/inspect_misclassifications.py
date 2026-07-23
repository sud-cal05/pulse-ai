"""
Diagnostic companion to run_evaluation.py: prints the FULL TEXT of every
misclassified golden item, not just the aggregate confusion-matrix counts.

Usage (requires a real OPENAI_API_KEY):
    PYTHONPATH=src python tests/golden/inspect_misclassifications.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Need BOTH the project root (so `tests.golden.*` imports resolve) and
# src/ (so `pulseai.*` imports resolve) on the path. Running this file
# directly (not via pytest) doesn't get pytest's automatic rootdir
# insertion, so both have to be added explicitly here.
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pulseai.ai.llm_client import LLMClient
from pulseai.core.analyze import analyze_feedback_item
from pulseai.schemas.feedback import FeedbackRecord
from tests.golden.run_evaluation import load_golden_set

def main() -> None:
    client = LLMClient()
    golden_items = load_golden_set()

    print(f"Checking {len(golden_items)} golden items for misclassifications...\n")
    mismatches = []

    for item in golden_items:
        record = FeedbackRecord(
            feedback_id=f"golden_{hash(item['feedback_text']) & 0xffffffff:08x}",
            source="manual",
            customer_id=None,
            raw_text=item["feedback_text"],
            cleaned_text=item["feedback_text"],
            language="en",
            created_at=datetime.now(),
        )
        result = analyze_feedback_item(record, client)

        category_wrong = result.primary_category.value != item["expected_category"]
        sentiment_wrong = result.sentiment.value != item["expected_sentiment"]
        urgency_wrong = result.urgency.value != item["expected_urgency"]

        if category_wrong or sentiment_wrong or urgency_wrong:
            mismatches.append({
                "text": item["feedback_text"],
                "expected_category": item["expected_category"],
                "predicted_category": result.primary_category.value,
                "category_wrong": category_wrong,
                "category_confidence": result.category_confidence,
                "expected_sentiment": item["expected_sentiment"],
                "predicted_sentiment": result.sentiment.value,
                "sentiment_wrong": sentiment_wrong,
                "expected_urgency": item["expected_urgency"],
                "predicted_urgency": result.urgency.value,
                "urgency_wrong": urgency_wrong,
            })

    if not mismatches:
        print("No misclassifications this run.")
        return

    print(f"Found {len(mismatches)} item(s) with at least one wrong dimension:\n")
    for i, m in enumerate(mismatches, 1):
        print(f"{i}. \"{m['text']}\"")
        if m["category_wrong"]:
            print(f"   CATEGORY: expected={m['expected_category']!r} "
                  f"predicted={m['predicted_category']!r} "
                  f"(confidence={m['category_confidence']:.2f})")
        if m["sentiment_wrong"]:
            print(f"   SENTIMENT: expected={m['expected_sentiment']!r} "
                  f"predicted={m['predicted_sentiment']!r}")
        if m["urgency_wrong"]:
            print(f"   URGENCY: expected={m['expected_urgency']!r} "
                  f"predicted={m['predicted_urgency']!r}")
        print()


if __name__ == "__main__":
    main()
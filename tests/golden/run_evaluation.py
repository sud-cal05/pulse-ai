"""
Runs the golden dataset against a REAL LLMClient and prints the full
evaluation report. This is the one file in tests/golden/ that actually
calls the model - everything else is pure math, testable without
network access.

Usage (with a real OPENAI_API_KEY configured):
    PYTHONPATH=src python tests/golden/run_evaluation.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pulseai.ai.llm_client import LLMClient
from pulseai.core.analyze import analyze_feedback_item
from pulseai.schemas.feedback import FeedbackRecord
from tests.golden.scoring import (
    EvaluationReport,
    build_confusion_matrix,
    compute_accuracy,
    compute_calibration_buckets,
)

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def load_golden_set() -> list[dict]:
    return json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))


def run_evaluation(client: LLMClient | None = None) -> EvaluationReport:
    client = client or LLMClient()
    golden_items = load_golden_set()

    category_predictions, category_truth = [], []
    sentiment_predictions, sentiment_truth = [], []
    urgency_predictions, urgency_truth = [], []
    confidences, category_correct = [], []

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

        category_predictions.append(result.primary_category.value)
        category_truth.append(item["expected_category"])
        sentiment_predictions.append(result.sentiment.value)
        sentiment_truth.append(item["expected_sentiment"])
        urgency_predictions.append(result.urgency.value)
        urgency_truth.append(item["expected_urgency"])

        confidences.append(result.category_confidence)
        category_correct.append(result.primary_category.value == item["expected_category"])

    return EvaluationReport(
        category_accuracy=compute_accuracy(category_predictions, category_truth),
        sentiment_accuracy=compute_accuracy(sentiment_predictions, sentiment_truth),
        urgency_accuracy=compute_accuracy(urgency_predictions, urgency_truth),
        category_confusion=build_confusion_matrix(category_predictions, category_truth),
        calibration_buckets=compute_calibration_buckets(confidences, category_correct),
        n_items=len(golden_items),
    )


if __name__ == "__main__":
    report = run_evaluation()
    print(report.summary())
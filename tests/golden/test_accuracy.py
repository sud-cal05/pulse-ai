"""
Runs the golden set against the REAL model and asserts minimum accuracy.
Skips automatically without a live OPENAI_API_KEY.

Run explicitly with:
    PYTHONPATH=src python -m pytest tests/golden/test_accuracy.py -v -s
"""

from __future__ import annotations

import os

import pytest

from tests.golden.run_evaluation import run_evaluation

MIN_CATEGORY_ACCURACY = 0.70
MIN_SENTIMENT_ACCURACY = 0.70
MIN_URGENCY_ACCURACY = 0.60


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Golden-set evaluation requires a live OPENAI_API_KEY; skipped in offline test runs.",
)
def test_golden_set_accuracy_meets_minimum_thresholds():
    report = run_evaluation()

    print("\n" + report.summary())

    assert report.category_accuracy >= MIN_CATEGORY_ACCURACY, (
        f"Category accuracy {report.category_accuracy:.1%} fell below the "
        f"{MIN_CATEGORY_ACCURACY:.0%} floor - check the confusion matrix "
        f"above to see which categories are being confused."
    )
    assert report.sentiment_accuracy >= MIN_SENTIMENT_ACCURACY, (
        f"Sentiment accuracy {report.sentiment_accuracy:.1%} fell below "
        f"the {MIN_SENTIMENT_ACCURACY:.0%} floor."
    )
    assert report.urgency_accuracy >= MIN_URGENCY_ACCURACY, (
        f"Urgency accuracy {report.urgency_accuracy:.1%} fell below the "
        f"{MIN_URGENCY_ACCURACY:.0%} floor."
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
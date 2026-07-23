"""
Tests for the evaluation scoring math itself - synthetic data, NOT a
real model. Proves the accuracy/confusion-matrix/calibration ARITHMETIC
is correct before ever trusting it to judge real model output.
"""

from __future__ import annotations

import pytest

from tests.golden.scoring import (
    build_confusion_matrix,
    compute_accuracy,
    compute_calibration_buckets,
)


def test_compute_accuracy_all_correct():
    assert compute_accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_compute_accuracy_partial():
    assert compute_accuracy(["a", "x", "c", "y"], ["a", "b", "c", "d"]) == 0.5


def test_compute_accuracy_none_correct():
    assert compute_accuracy(["x", "y"], ["a", "b"]) == 0.0


def test_compute_accuracy_empty_lists():
    assert compute_accuracy([], []) == 0.0


def test_compute_accuracy_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        compute_accuracy(["a", "b"], ["a"])


def test_confusion_matrix_tallies_misclassifications():
    predictions = ["bug_defect", "performance", "bug_defect", "billing_pricing"]
    ground_truth = ["performance", "performance", "bug_defect", "billing_pricing"]
    matrix = build_confusion_matrix(predictions, ground_truth)

    assert matrix[("performance", "bug_defect")] == 1
    assert matrix[("performance", "performance")] == 1
    assert matrix[("bug_defect", "bug_defect")] == 1
    assert matrix[("billing_pricing", "billing_pricing")] == 1


def test_calibration_buckets_group_correctly():
    confidences = [0.95, 0.90, 0.60, 0.40, 0.75]
    correct =     [True,  False, True, False, True]

    buckets = compute_calibration_buckets(confidences, correct)
    by_label = {b.label: b for b in buckets}

    assert by_label["0.85-1.00"].count == 2
    assert by_label["0.85-1.00"].correct == 1
    assert by_label["0.85-1.00"].accuracy == pytest.approx(0.5)

    assert by_label["0.50-0.70"].count == 1
    assert by_label["0.50-0.70"].correct == 1

    assert by_label["0.00-0.50"].count == 1
    assert by_label["0.00-0.50"].correct == 0


def test_calibration_reveals_overconfidence():
    confidences = [0.95] * 10
    correct = [True, False, True, False, True, False, True, False, True, False]

    buckets = compute_calibration_buckets(confidences, correct)
    top_bucket = next(b for b in buckets if b.label == "0.85-1.00")

    assert top_bucket.count == 10
    assert top_bucket.accuracy == pytest.approx(0.5)


def test_calibration_empty_bucket_has_zero_accuracy_not_crash():
    buckets = compute_calibration_buckets([0.95], [True])
    empty_bucket = next(b for b in buckets if b.label == "0.00-0.50")
    assert empty_bucket.count == 0
    assert empty_bucket.accuracy == 0.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
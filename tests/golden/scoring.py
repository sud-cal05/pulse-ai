"""
Pure evaluation math for the golden dataset. Zero dependency on any LLM
client - every function here takes plain lists of predictions and
ground truth, so the scoring logic itself is unit-testable with
synthetic data, independent of model quality or API access.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


def compute_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions ({len(predictions)}) and ground_truth "
            f"({len(ground_truth)}) must be the same length"
        )
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions)


def build_confusion_matrix(
    predictions: list[str], ground_truth: list[str]
) -> dict[tuple[str, str], int]:
    matrix: Counter[tuple[str, str]] = Counter()
    for pred, actual in zip(predictions, ground_truth):
        matrix[(actual, pred)] += 1
    return dict(matrix)


def format_confusion_matrix(matrix: dict[tuple[str, str], int]) -> str:
    lines = []
    for (expected, predicted), count in sorted(matrix.items(), key=lambda kv: -kv[1]):
        if expected != predicted:
            lines.append(f"  expected={expected!r:30s} predicted={predicted!r:30s} count={count}")
    if not lines:
        return "  (no misclassifications)"
    return "\n".join(lines)


@dataclass
class CalibrationBucket:
    label: str
    count: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.count if self.count > 0 else 0.0


DEFAULT_BUCKET_EDGES = [0.0, 0.5, 0.7, 0.85, 1.01]


def compute_calibration_buckets(
    confidences: list[float],
    correct: list[bool],
    bucket_edges: list[float] = DEFAULT_BUCKET_EDGES,
) -> list[CalibrationBucket]:
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")

    buckets = []
    for i in range(len(bucket_edges) - 1):
        low, high = bucket_edges[i], bucket_edges[i + 1]
        label = f"{low:.2f}-{high:.2f}" if high <= 1.0 else f"{low:.2f}-1.00"
        buckets.append(CalibrationBucket(label=label))

    for conf, was_correct in zip(confidences, correct):
        for i in range(len(bucket_edges) - 1):
            low, high = bucket_edges[i], bucket_edges[i + 1]
            if low <= conf < high:
                buckets[i].count += 1
                if was_correct:
                    buckets[i].correct += 1
                break

    return buckets


def format_calibration_table(buckets: list[CalibrationBucket]) -> str:
    lines = ["  confidence range | n items | actual accuracy"]
    for b in buckets:
        if b.count == 0:
            lines.append(f"  {b.label:17s} | {b.count:7d} | (no items)")
        else:
            lines.append(f"  {b.label:17s} | {b.count:7d} | {b.accuracy:.0%}")
    return "\n".join(lines)


@dataclass
class EvaluationReport:
    category_accuracy: float
    sentiment_accuracy: float
    urgency_accuracy: float
    category_confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    calibration_buckets: list[CalibrationBucket] = field(default_factory=list)
    n_items: int = 0

    def summary(self) -> str:
        lines = [
            f"Golden set evaluation - {self.n_items} items",
            f"  Category accuracy:  {self.category_accuracy:.1%}",
            f"  Sentiment accuracy: {self.sentiment_accuracy:.1%}",
            f"  Urgency accuracy:   {self.urgency_accuracy:.1%}",
            "",
            "Category confusion matrix (misclassifications only):",
            format_confusion_matrix(self.category_confusion),
            "",
            "Confidence calibration (category_confidence vs actual accuracy):",
            format_calibration_table(self.calibration_buckets),
        ]
        return "\n".join(lines)
"""
Phase 7 tests: deterministic clustering, cohesion, LLM labeling with
retry/fallback, and full end-to-end theme extraction.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from pulseai.core.themes import (
    _cohesion,
    _label_cluster_with_llm,
    build_theme_clusters,
    cluster_embeddings,
)
from pulseai.schemas.analysis import Category, PerItemAnalysis, Sentiment, Urgency
from pulseai.schemas.themes import ThemeLabelOutput


class _FakeConfig:
    model = "fake-model-v1"


class FakeLLMClient:
    def __init__(self, responses: list):
        self._responses = list(responses)
        self.config = _FakeConfig()
        self.call_count = 0

    def complete_messages(self, messages: list[dict]) -> str:
        self.call_count += 1
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeEmbeddingClient:
    def __init__(self, embeddings: list[list[float]]):
        self._embeddings = embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == len(self._embeddings)
        return self._embeddings


def _analysis(
    feedback_id: str,
    themes: list[str],
    quote: str,
    sentiment_score: float = -0.5,
    urgency_score: float = 0.5,
) -> PerItemAnalysis:
    return PerItemAnalysis(
        feedback_id=feedback_id,
        primary_category=Category.PERFORMANCE,
        category_confidence=0.9,
        sentiment=Sentiment.NEGATIVE,
        sentiment_score=sentiment_score,
        sentiment_confidence=0.9,
        urgency=Urgency.MEDIUM,
        urgency_score=urgency_score,
        item_themes=themes,
        key_quote=quote,
        model_version="fake-model-v1",
        prompt_version="analysis_v1",
        analyzed_at=datetime(2026, 7, 20, 9, 0),
    )


def test_two_clear_clusters_separate_correctly():
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.95, 0.05, 0.0],
        [0.0, 1.0, 0.0],
        [0.05, 0.95, 0.0],
    ]
    clusters = cluster_embeddings(embeddings, distance_threshold=0.35)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 2]


def test_clustering_is_deterministic_across_repeated_calls():
    embeddings = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.5, 0.5]]
    result_a = cluster_embeddings(embeddings, distance_threshold=0.3)
    result_b = cluster_embeddings(embeddings, distance_threshold=0.3)
    normalize = lambda clusters: sorted(tuple(sorted(c)) for c in clusters)
    assert normalize(result_a) == normalize(result_b)


def test_single_item_returns_single_cluster():
    assert cluster_embeddings([[1.0, 0.0, 0.0]]) == [[0]]


def test_empty_input_returns_no_clusters():
    assert cluster_embeddings([]) == []


def test_cohesion_identical_vectors_is_perfect():
    identical = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert _cohesion(identical) == pytest.approx(1.0)


def test_cohesion_orthogonal_vectors_is_low():
    orthogonal = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert _cohesion(orthogonal) == pytest.approx(0.0, abs=1e-6)


def test_cohesion_singleton_is_defined_as_perfect():
    assert _cohesion(np.array([[1.0, 0.0, 0.0]])) == 1.0


VALID_LABEL_RESPONSE = json.dumps({
    "label": "checkout timeout on mobile",
    "description": "Customers report checkout failing to complete on mobile devices.",
})


def test_label_cluster_success():
    members = [_analysis("fb_1", ["checkout timeout"], "checkout hangs on mobile")]
    client = FakeLLMClient([VALID_LABEL_RESPONSE])
    result = _label_cluster_with_llm(members, client)
    assert result.label == "checkout timeout on mobile"


def test_label_cluster_retries_then_succeeds():
    members = [_analysis("fb_1", ["checkout timeout"], "checkout hangs on mobile")]
    client = FakeLLMClient(["not json", VALID_LABEL_RESPONSE])
    result = _label_cluster_with_llm(members, client)
    assert client.call_count == 2
    assert result.label == "checkout timeout on mobile"


def test_label_cluster_falls_back_without_crashing():
    members = [_analysis("fb_1", ["checkout timeout"], "checkout hangs on mobile")]
    client = FakeLLMClient(["garbage", "still garbage"])
    result = _label_cluster_with_llm(members, client)
    assert isinstance(result, ThemeLabelOutput)
    assert result.label == "checkout timeout"


def test_build_theme_clusters_end_to_end():
    analyses = [
        _analysis("fb_1", ["checkout timeout"], "checkout hangs on mobile",
                   sentiment_score=-0.8, urgency_score=0.9),
        _analysis("fb_2", ["checkout timeout"], "checkout freezes on phone",
                   sentiment_score=-0.7, urgency_score=0.8),
        _analysis("fb_3", ["slow search"], "search takes forever",
                   sentiment_score=-0.2, urgency_score=0.2),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.97, 0.03, 0.0],
        [0.0, 0.0, 1.0],
    ]
    embed_client = FakeEmbeddingClient(embeddings)
    llm_client = FakeLLMClient([
        VALID_LABEL_RESPONSE,
        json.dumps({"label": "slow search results", "description": "Search is slow."}),
    ])

    clusters = build_theme_clusters(
        analyses, embed_client, llm_client, period="2026-W29", distance_threshold=0.3
    )

    assert len(clusters) == 2
    sizes = sorted(c.size for c in clusters)
    assert sizes == [1, 2]

    top = clusters[0]
    assert top.label == "checkout timeout on mobile"
    assert top.priority_score == pytest.approx(1.0)
    assert set(top.member_feedback_ids) == {"fb_1", "fb_2"}


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

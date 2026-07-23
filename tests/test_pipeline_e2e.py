"""
Phase 10 backend test: run_full_pipeline ties all 9 stages together into
a DashboardPayload. Tested here with fake clients (no network) using the
real sample CSV, so the dashboard itself can stay a thin, mostly-untested
rendering layer.
"""

from __future__ import annotations

import json

import pytest

from pulseai.core.pipeline import run_full_pipeline
from pulseai.schemas.report import DashboardPayload
from pulseai.storage.db import get_connection


class _FakeConfig:
    model = "fake-model-v1"
    embedding_model = "fake-embedding-v1"


class FakeLLMClient:
    def __init__(self):
        self.config = _FakeConfig()
        self.call_count = 0

    def complete_messages(self, messages: list[dict]) -> str:
        self.call_count += 1
        system_content = messages[0]["content"]

        if "customer experience" in system_content.lower() and "labeling" in system_content.lower():
            return json.dumps({"label": "generic issue", "description": "A generic issue."})
        if "executive brief" in system_content.lower() or "VP of Customer Experience" in system_content:
            return json.dumps({
                "headline": "Test headline.",
                "key_insights": [],
                "recommended_actions": [],
                "watch_items": [],
                "caveats": "Test run with fake data.",
            })
        return json.dumps({
            "primary_category": "performance",
            "secondary_categories": [],
            "category_confidence": 0.9,
            "sentiment": "negative",
            "sentiment_score": -0.5,
            "sentiment_confidence": 0.9,
            "urgency": "medium",
            "urgency_score": 0.5,
            "urgency_signals": ["test signal"],
            "item_themes": ["test theme"],
            "key_quote": "test quote",
        })


class FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(i), float(i) * 0.1, 1.0] for i, _ in enumerate(texts)]


def test_run_full_pipeline_end_to_end_on_real_sample_csv(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    llm_client = FakeLLMClient()
    embed_client = FakeEmbeddingClient()

    payload = run_full_pipeline(
        "sample_data/feedback_sample.csv", llm_client, embed_client, conn
    )

    assert isinstance(payload, DashboardPayload)
    assert payload.aggregate.total_feedback > 0
    assert len(payload.items) == payload.aggregate.total_feedback
    assert payload.summary.headline == "Test headline."
    assert payload.summary.caveats == "Test run with fake data."


def test_run_full_pipeline_second_run_uses_cache(tmp_path):
    conn = get_connection(tmp_path / "test.db")
    llm_client = FakeLLMClient()
    embed_client = FakeEmbeddingClient()

    run_full_pipeline("sample_data/feedback_sample.csv", llm_client, embed_client, conn)
    calls_after_first_run = llm_client.call_count

    run_full_pipeline("sample_data/feedback_sample.csv", llm_client, embed_client, conn)
    calls_after_second_run = llm_client.call_count

    assert calls_after_second_run < calls_after_first_run * 2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
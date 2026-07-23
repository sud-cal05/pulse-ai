"""
EmbeddingClient - the one place that calls the embedding endpoint
(DESIGN.md section 5.2, section 12). Kept separate from LLMClient (chat
completions) because it's a genuinely different API surface and a
different job: this produces vectors for semantic clustering, not JSON
analysis.
"""

from __future__ import annotations

import logging

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from pulseai.ai.config import LLMConfig, load_llm_config

logger = logging.getLogger("pulseai.embeddings")


class EmbeddingCallFailed(Exception):
    """Raised when the embeddings call fails after retries are exhausted."""


class EmbeddingClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or load_llm_config()
        self._client = OpenAI(api_key=self.config.api_key)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        stop=stop_after_attempt(2),
        wait=wait_fixed(1),
        reraise=True,
    )
    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self.config.embedding_model, input=texts
        )
        return [item.embedding for item in response.data]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batched embedding call: one request for the whole list."""
        if not texts:
            return []
        try:
            return self._embed(texts)
        except (RateLimitError, APITimeoutError, APIError) as exc:
            logger.error("Embedding call failed after retry: %s", exc)
            raise EmbeddingCallFailed(str(exc)) from exc

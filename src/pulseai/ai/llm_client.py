"""
LLMClient — the ONE place in the codebase that talks to the model provider.

Why this abstraction exists (DESIGN.md §5.1, §9):
- Every other module (analyze.py, themes.py, summarize.py) calls `LLMClient`,
  never the OpenAI SDK directly. Swapping providers, or A/B testing
  gpt-5-nano vs gpt-4.1-mini, is a change in exactly one file.
- Determinism is enforced here, once, so no call site can accidentally
  drift from it: reasoning models get `reasoning_effort`, non-reasoning
  models get `temperature=0`. Either way the same input should reliably
  produce the same output (rubric M5B1 / M5S2).
- Retry-then-fallback lives here so every caller gets it for free
  (rubric M5B3 — API failures must never crash the pipeline).

Phase 1 scope: just prove we can make ONE successful call and get text
back. Structured-output (schema-constrained JSON) is wired in at Phase 5
once the Pydantic schemas exist (see schemas/analysis.py).
"""

from __future__ import annotations

import logging

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from pulseai.ai.config import LLMConfig, load_llm_config

logger = logging.getLogger("pulseai.llm_client")


class LLMCallFailed(Exception):
    """Raised when the provider call fails after retries are exhausted.
    Callers (e.g. analyze.py) catch this and apply the safe fallback
    described in DESIGN.md §9 / §18 — never let this propagate into a crash.
    """


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or load_llm_config()
        self._client = OpenAI(api_key=self.config.api_key)

    @retry(
        # Only retry on transient/network-ish failures — a genuine bad
        # request (e.g. malformed schema) would just fail the same way
        # again, so we don't waste a retry on it.
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        stop=stop_after_attempt(2),  # one retry, per DESIGN.md §9
        wait=wait_fixed(1),
        reraise=True,
    )
    def _call(self, messages: list[dict]) -> str:
        kwargs = dict(model=self.config.model, messages=messages)

        if self.config.is_reasoning_model:
            kwargs["reasoning_effort"] = self.config.reasoning_effort
        else:
            kwargs["temperature"] = self.config.temperature

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """
        Plain text completion. Used for Phase 1's connectivity smoke test
        and anywhere a full JSON schema isn't needed yet.

        On failure (after the built-in retry): raises LLMCallFailed rather
        than letting the raw provider exception surface, so every call
        site has one exception type to catch (DESIGN.md §9, §18).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return self._call(messages)
        except (RateLimitError, APITimeoutError, APIError) as exc:
            logger.error("LLM call failed after retry: %s", exc)
            raise LLMCallFailed(str(exc)) from exc


if __name__ == "__main__":
    # Phase 1 acceptance test: prove the key loads and one call succeeds.
    # Run with: python -m pulseai.ai.llm_client
    client = LLMClient()
    print(f"Using model: {client.config.model} "
          f"(reasoning model: {client.config.is_reasoning_model})")
    try:
        reply = client.complete(
            system_prompt="You are a terse test assistant.",
            user_prompt="Reply with exactly: PulseAI connection OK",
        )
        print("Model replied:", reply)
    except LLMCallFailed as e:
        print("Call failed (this is the graceful path, not a crash):", e)

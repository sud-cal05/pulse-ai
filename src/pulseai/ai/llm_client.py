"""
LLMClient - the ONE place in the codebase that talks to the model provider.

Why this abstraction exists (DESIGN.md section 5.1, section 9):
- Every other module (analyze.py, themes.py, summarize.py) calls LLMClient,
  never the OpenAI SDK directly. Swapping providers, or A/B testing
  gpt-5-nano vs gpt-4.1-mini, is a change in exactly one file.
- Determinism is enforced here, once, so no call site can accidentally
  drift from it: reasoning models get reasoning_effort, non-reasoning
  models get temperature=0.
- Retry-then-fallback lives here so every caller gets it for free
  (rubric M5B3 - API failures must never crash the pipeline).
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
    described in DESIGN.md section 9 / section 18 - never let this propagate
    into a crash."""


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or load_llm_config()
        self._client = OpenAI(api_key=self.config.api_key)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
        stop=stop_after_attempt(2),
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

    def complete_messages(self, messages: list[dict]) -> str:
        """
        Same contract as complete(), but for a full multi-turn message
        list (system + few-shot example turns + the real input) rather
        than a single system/user pair. This is what analyze.py and the
        zero-shot probe use, since both need the few-shot-boosted prompt
        from prompt_loader.build_analysis_messages().
        """
        try:
            return self._call(messages)
        except (RateLimitError, APITimeoutError, APIError) as exc:
            logger.error("LLM call failed after retry: %s", exc)
            raise LLMCallFailed(str(exc)) from exc


if __name__ == "__main__":
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

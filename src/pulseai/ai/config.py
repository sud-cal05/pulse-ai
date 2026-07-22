"""
Centralized configuration, loaded from environment variables (.env).

Why this file exists (see DESIGN.md §5.1, §9):
- Every knob that affects determinism (model, temperature, reasoning effort)
  lives in exactly one place, so changing it is a one-line config edit,
  not a hunt through the codebase.
- Nothing here is a secret value — only the *names* of env vars are
  hardcoded. The actual key is read at runtime and never logged or printed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env once, at import time. In production (Docker/Streamlit Cloud)
# real env vars are injected directly and this becomes a no-op.
load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    model: str
    reasoning_effort: str
    temperature: float
    embedding_model: str

    @property
    def is_reasoning_model(self) -> bool:
        """
        Reasoning models (the GPT-5 family) use `reasoning_effort` to control
        determinism/quality instead of `temperature`. Non-reasoning models
        (e.g. gpt-4.1-mini) use `temperature` instead. We branch on this in
        llm_client.py so the same LLMConfig works for either model family —
        this is the whole reason the provider is swappable in one file.
        """
        return self.model.startswith("gpt-5")


def load_llm_config() -> LLMConfig:
    """
    Reads and validates required config from the environment.
    Raises a clear error immediately if a required var is missing —
    fail fast at startup, not three stages deep in the pipeline.
    """
    provider = os.getenv("LLM_PROVIDER", "openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
            "your key. Never hardcode it in source."
        )

    return LLMConfig(
        provider=provider,
        api_key=api_key,
        model=os.getenv("LLM_MODEL", "gpt-5-nano"),
        reasoning_effort=os.getenv("LLM_REASONING_EFFORT", "low"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )

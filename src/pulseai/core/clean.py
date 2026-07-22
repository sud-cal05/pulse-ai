"""
Stage 3 — Cleaning (DESIGN.md §6, §16).

Two jobs: (1) normalize noisy text before it reaches the LLM, and
(2) detect language so non-English feedback is flagged rather than
silently mis-analyzed by an English-tuned prompt.

Cleaning is deliberately CONSERVATIVE (DESIGN.md §6 stage 3 failure case:
"over-aggressive cleaning strips meaning"). We strip formatting noise —
whitespace, quoted email replies — but never touch wording, punctuation
choices, or casing, since sentiment/urgency signals can live in those.
"""

from __future__ import annotations

import re

from langdetect import LangDetectException, detect

# Matches common email-reply quoting: "> some previous text" or
# "On Jul 20, 2026, Jane wrote:" style headers.
_QUOTE_LINE_RE = re.compile(r"^\s*>.*$", re.MULTILINE)
_EMAIL_HEADER_RE = re.compile(
    r"^On .{0,80}wrote:\s*$", re.MULTILINE | re.IGNORECASE
)
_WHITESPACE_RE = re.compile(r"\s+")

# A string with no letters or digits at all — used to catch emoji-only
# or punctuation-only feedback (DESIGN.md §16 edge case).
_HAS_ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)


def clean_text(raw_text: str) -> str:
    """Strip quoting/signature noise and normalize whitespace. Does NOT
    alter wording, casing, or punctuation — those carry sentiment signal."""
    text = _EMAIL_HEADER_RE.sub("", raw_text)
    text = _QUOTE_LINE_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_emoji_only(text: str) -> bool:
    """
    True if the text has no letters or digits at all (pure emoji/symbols).
    This is one of the explicit edge cases in DESIGN.md §16: such feedback
    carries real sentiment but no words a language model can "read" in the
    usual sense, so it gets flagged rather than silently misclassified.
    """
    return not _HAS_ALNUM_RE.search(text)


def detect_language(text: str) -> str:
    """
    Best-effort ISO 639-1 language code, or 'unknown'.

    langdetect throws on very short or non-alphabetic input (emoji-only,
    single words) rather than returning a low-confidence guess — we catch
    that explicitly rather than letting it propagate, since a language-
    detection hiccup should never crash the pipeline (same graceful-
    degradation principle as everywhere else in this stage).
    """
    if is_emoji_only(text) or len(text.strip()) < 3:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
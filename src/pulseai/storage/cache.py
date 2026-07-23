"""
Cache-aware batch analysis (DESIGN.md section 6 stage 9, section 9, section 13).

This is the function the real pipeline calls instead of analyze_batch
directly: for each record, check the cache FIRST; only call the LLM on
a genuine miss.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from pulseai.ai.llm_client import LLMClient
from pulseai.ai.prompt_loader import SYSTEM_PROMPT_VERSION
from pulseai.core.analyze import DEFAULT_CONFIDENCE_THRESHOLD, analyze_feedback_item
from pulseai.schemas.analysis import PerItemAnalysis
from pulseai.schemas.feedback import FeedbackRecord
from pulseai.storage.db import get_cached_analysis, save_analysis, upsert_feedback_record

logger = logging.getLogger("pulseai.cache")


@dataclass
class CacheStats:
    total: int = 0
    hits: int = 0
    misses: int = 0


def analyze_batch_cached(
    records: list[FeedbackRecord],
    client: LLMClient,
    conn: sqlite3.Connection,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[list[PerItemAnalysis], CacheStats]:
    stats = CacheStats(total=len(records))
    results: list[PerItemAnalysis] = []

    for record in records:
        upsert_feedback_record(conn, record)

        cached = get_cached_analysis(
            conn, record.feedback_id, SYSTEM_PROMPT_VERSION, client.config.model
        )
        if cached is not None:
            stats.hits += 1
            results.append(cached)
            continue

        stats.misses += 1
        analysis = analyze_feedback_item(record, client, confidence_threshold)
        save_analysis(conn, analysis)
        results.append(analysis)

    logger.info(
        "Batch analysis: %d total, %d cache hits, %d cache misses (LLM calls)",
        stats.total, stats.hits, stats.misses,
    )
    return results, stats

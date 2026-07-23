"""
Stage 6 - Theme extraction (DESIGN.md section 5.2, section 6, section 11).

Clustering (pure code, deterministic): group semantically similar items
using embeddings + AgglomerativeClustering - chosen because it has no
random initialization, unlike k-means, which matters for the same
consistency reasons as temperature=0 (rubric M5B1/M5S2).

Labeling (LLM, per cluster): given a cluster's actual members' themes
and quotes, produce ONE specific label + description. Counts, cohesion,
and priority scores are all computed in code, never by the LLM.

No vector database is used: clustering happens entirely in memory,
appropriate for a corpus that fits in RAM.
"""

from __future__ import annotations

import json
import logging

import numpy as np
from pydantic import ValidationError
from sklearn.cluster import AgglomerativeClustering

from pulseai.ai.embeddings import EmbeddingCallFailed, EmbeddingClient
from pulseai.ai.llm_client import LLMCallFailed, LLMClient
from pulseai.ai.prompt_loader import build_theme_labeling_messages
from pulseai.schemas.analysis import PerItemAnalysis
from pulseai.schemas.themes import RepresentativeQuote, ThemeCluster, ThemeLabelOutput

logger = logging.getLogger("pulseai.themes")

DEFAULT_DISTANCE_THRESHOLD = 0.35
MAX_MEMBERS_FOR_LABELING = 8
MAX_REPRESENTATIVE_QUOTES = 3


def _cluster_text(analysis: PerItemAnalysis) -> str:
    if analysis.item_themes:
        return "; ".join(analysis.item_themes)
    return analysis.key_quote


def cluster_embeddings(
    embeddings: list[list[float]],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> list[list[int]]:
    n = len(embeddings)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    X = np.array(embeddings)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(X)

    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)
    return list(clusters.values())


def _cohesion(embeddings_subset: np.ndarray) -> float:
    n = len(embeddings_subset)
    if n <= 1:
        return 1.0
    normed = embeddings_subset / np.linalg.norm(embeddings_subset, axis=1, keepdims=True)
    sim_matrix = normed @ normed.T
    off_diagonal_sum = sim_matrix.sum() - n
    pair_count = n * (n - 1)
    return float(off_diagonal_sum / pair_count)


def _label_cluster_fallback(members: list[PerItemAnalysis]) -> ThemeLabelOutput:
    all_themes = [t for m in members for t in m.item_themes]
    fallback_label = all_themes[0] if all_themes else members[0].key_quote[:60]
    logger.warning(
        "Falling back to heuristic label for a cluster of %d items after "
        "repeated labeling failure.", len(members),
    )
    return ThemeLabelOutput(
        label=fallback_label,
        description="Automated labeling failed for this cluster; label "
        "derived from raw item themes and should be reviewed.",
    )


def _label_cluster_with_llm(members: list[PerItemAnalysis], client: LLMClient) -> ThemeLabelOutput:
    sample = members[:MAX_MEMBERS_FOR_LABELING]
    cluster_themes = [m.item_themes for m in sample]
    cluster_quotes = [m.key_quote for m in sample]
    messages = build_theme_labeling_messages(cluster_themes, cluster_quotes)

    for attempt in (1, 2):
        try:
            raw = client.complete_messages(messages)
            parsed = json.loads(raw)
            return ThemeLabelOutput(**parsed)
        except LLMCallFailed as exc:
            logger.warning("Labeling attempt %d/2 failed: LLM call failed (%s)", attempt, exc)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Labeling attempt %d/2 failed: invalid response (%s)", attempt, exc)

    return _label_cluster_fallback(members)


def build_theme_clusters(
    analyses: list[PerItemAnalysis],
    embedding_client: EmbeddingClient,
    llm_client: LLMClient,
    period: str,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> list[ThemeCluster]:
    if not analyses:
        return []

    texts = [_cluster_text(a) for a in analyses]
    try:
        embeddings = embedding_client.embed_texts(texts)
    except EmbeddingCallFailed as exc:
        logger.error("Embedding call failed for the whole batch: %s", exc)
        raise

    index_clusters = cluster_embeddings(embeddings, distance_threshold)
    embeddings_arr = np.array(embeddings)

    raw_clusters: list[dict] = []
    for indices in index_clusters:
        members = [analyses[i] for i in indices]
        member_embeddings = embeddings_arr[indices]

        avg_sentiment = float(np.mean([m.sentiment_score for m in members]))
        avg_urgency = float(np.mean([m.urgency_score for m in members]))
        cohesion = _cohesion(member_embeddings)

        volume_share = len(members) / len(analyses)
        negativity_weight = (1.0 - avg_sentiment) / 2.0
        raw_priority = volume_share * avg_urgency * negativity_weight

        label_output = _label_cluster_with_llm(members, llm_client)

        quotes = []
        seen_quotes = set()
        for m in members:
            if m.key_quote not in seen_quotes:
                quotes.append(RepresentativeQuote(feedback_id=m.feedback_id, text=m.key_quote))
                seen_quotes.add(m.key_quote)
            if len(quotes) >= MAX_REPRESENTATIVE_QUOTES:
                break

        raw_clusters.append({
            "label": label_output.label,
            "description": label_output.description,
            "member_feedback_ids": [m.feedback_id for m in members],
            "size": len(members),
            "avg_sentiment_score": avg_sentiment,
            "avg_urgency_score": avg_urgency,
            "representative_quotes": quotes,
            "cohesion": cohesion,
            "raw_priority": raw_priority,
        })

    max_raw_priority = max((c["raw_priority"] for c in raw_clusters), default=0.0)

    theme_clusters = []
    for i, c in enumerate(raw_clusters):
        normalized_priority = (
            c["raw_priority"] / max_raw_priority if max_raw_priority > 0 else 0.0
        )
        theme_clusters.append(
            ThemeCluster(
                theme_id=f"theme_{i:03d}",
                label=c["label"],
                description=c["description"],
                member_feedback_ids=c["member_feedback_ids"],
                size=c["size"],
                avg_sentiment_score=c["avg_sentiment_score"],
                avg_urgency_score=c["avg_urgency_score"],
                representative_quotes=c["representative_quotes"],
                cohesion=c["cohesion"],
                priority_score=normalized_priority,
                period=period,
            )
        )

    theme_clusters.sort(key=lambda t: t.priority_score, reverse=True)
    return theme_clusters

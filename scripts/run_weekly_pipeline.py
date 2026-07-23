from pulseai.ai.embeddings import EmbeddingClient
from pulseai.ai.llm_client import LLMClient
from pulseai.core.aggregate import build_weekly_aggregate, get_iso_week_bounds
from pulseai.core.pipeline import build_feedback_batch
from pulseai.core.themes import build_theme_clusters
from pulseai.storage.cache import analyze_batch_cached
from pulseai.storage.db import get_connection
from pulseai.core.summarize import generate_weekly_summary

records, ingest_stats = build_feedback_batch("sample_data/feedback_sample.csv")
print(ingest_stats)

llm_client = LLMClient()
embed_client = EmbeddingClient()
conn = get_connection("pulseai.db")

results, cache_stats = analyze_batch_cached(records, llm_client, conn)
print(cache_stats)

period_start, period_end, period_str = get_iso_week_bounds(records[0].created_at.date())
themes = build_theme_clusters(results, embed_client, llm_client, period=period_str)

aggregate = build_weekly_aggregate(results, themes, period_start, period_end)

print(f"\n=== Weekly Aggregate ({aggregate.period_start} to {aggregate.period_end}) ===")
print(f"Total feedback: {aggregate.total_feedback}")
print(f"Review queue: {aggregate.review_queue_count}")
print("\nCategory distribution:")
for cat, count in aggregate.category_distribution.items():
    print(f"  {cat.value}: {count}")
print("\nSentiment distribution:")
for sent, count in aggregate.sentiment_distribution.items():
    print(f"  {sent.value}: {count}")
print("\nTop themes:")
for t in aggregate.top_themes:
    print(f"  [{t.priority_score:.2f}] {t.label} (size={t.size})")

summary = generate_weekly_summary(aggregate, llm_client)

print(f"\n=== Executive Summary ===")
print(f"Headline: {summary.headline}\n")
print("Key Insights:")
for insight in summary.key_insights:
    print(f"  - {insight.statement}")
    print(f"    (cited: {insight.supporting_feedback_ids}, confidence={insight.confidence:.2f})")
print("\nRecommended Actions:")
for action in summary.recommended_actions:
    print(f"  [{action.priority}] {action.action}")
    print(f"      Rationale: {action.rationale}")
    print(f"      Expected impact: {action.expected_impact}")
if summary.watch_items:
    print("\nWatch items:")
    for item in summary.watch_items:
        print(f"  - {item}")
print(f"\nCaveats: {summary.caveats}")
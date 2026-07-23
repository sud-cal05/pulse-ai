from pulseai.ai.llm_client import LLMClient
from pulseai.core.pipeline import build_feedback_batch
from pulseai.storage.cache import analyze_batch_cached
from pulseai.storage.db import get_connection

records, _ = build_feedback_batch('sample_data/feedback_sample.csv')
client = LLMClient()
conn = get_connection('pulseai.db')
results, stats = analyze_batch_cached(records, client, conn)
print(stats)

for r, rec in zip(results, records):
    if r.needs_human_review:
        print(f"[REVIEW] {r.feedback_id} | cat_conf={r.category_confidence:.2f} sent_conf={r.sentiment_confidence:.2f} | text: {rec.cleaned_text[:60]!r}")
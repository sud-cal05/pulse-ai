# PulseAI — Customer Feedback Intelligence Platform

An AI-powered platform that classifies customer feedback, scores sentiment and urgency, discovers recurring themes, and generates a **weekly executive brief where every insight cites the real feedback behind it.**

Full architectural rationale, alternatives considered, and rubric mapping for every decision live in [`docs/DESIGN.md`](docs/DESIGN.md). This README covers what you need to run it and understand it.

---

## Table of Contents

- [PulseAI — Customer Feedback Intelligence Platform](#pulseai--customer-feedback-intelligence-platform)
  - [Table of Contents](#table-of-contents)
  - [The hero feature](#the-hero-feature)
  - [Who this is for](#who-this-is-for)
  - [Architecture](#architecture)
  - [Setup](#setup)
  - [Running it](#running-it)
  - [Screenshots](#screenshots)
  - [The AI pipeline](#the-ai-pipeline)
  - [Prompt strategy](#prompt-strategy)
  - [Key design decisions](#key-design-decisions)
  - [Testing](#testing)
  - [Known limitations](#known-limitations)
  - [Future improvements](#future-improvements)
  - [Lessons learned](#lessons-learned)
  - [Roadmap status](#roadmap-status)

---

## The hero feature

The weekly summary isn't free-form AI prose. It's a **Grounded Action Brief**: every insight cites the specific feedback IDs that support it, every action is ranked by a transparent Priority Score (`volume × urgency × negativity`, normalized), and every citation is validated after generation — any insight citing feedback the model wasn't actually shown gets silently dropped, never invented around.

This is deliberately an anti-hallucination mechanism, not just a formatting choice. See [`docs/DESIGN.md` §3](docs/DESIGN.md) for the full rationale, and `tests/test_summarize.py::test_insight_citing_fake_feedback_id_is_dropped` for proof it actually catches a fabricated citation.

## Who this is for

CX and Product teams drowning in feedback across channels. Instead of reading through hundreds of raw comments, a PM opens the weekly brief and knows the top three things to fix, how many customers each affects, and can click through to the real quotes behind every claim.

## Architecture

```
                    ┌──────────────────────────────┐
                    │   Streamlit Dashboard (UI)    │  ← thin, imports core
                    └───────────────┬──────────────┘
                                    │
        ┌───────────────────────────────────────────────┐
        │            CORE PIPELINE (pure Python)         │
        │  ingest → validate → clean → analyze (LLM) →   │
        │  cluster themes → aggregate → summarize (LLM)  │
        └───────────────┬───────────────────────────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
      ┌──────────────┐   ┌──────────────────┐
      │  SQLite DB    │   │  LLM provider API │
      │ (results +    │   │  (temp 0/low      │
      │  raw JSON)    │   │  reasoning effort, │
      │  content-hash │   │  pinned model)     │
      │  keyed cache) │   │                    │
      └──────────────┘   └──────────────────┘
```

The dashboard is a genuinely thin client: `dashboard/app.py` calls exactly one function, `run_full_pipeline()`, and renders whatever it returns. Every architectural decision — caching, retry/fallback, deterministic aggregation, grounded summarization — lives in `core/` and is completely decoupled from the UI. See `docs/DESIGN.md` §4 for why this matters.

## Setup

```bash
git clone <this-repo>
cd pulseai

python3 -m venv venv
source venv/bin/activate      # re-run this any time you open a new terminal

pip install -r requirements.txt

cp .env.example .env
# edit .env and add your real OPENAI_API_KEY
```

**Verify your connection:**
```bash
PYTHONPATH=src python -m pulseai.ai.llm_client
```
Expected output: `Model replied: PulseAI connection OK`

## Running it

**The dashboard (the main deliverable):**
```bash
streamlit run dashboard/app.py
```
Opens in your browser. Set a CSV path in the sidebar (defaults to the bundled sample data), click **Run weekly analysis**, and explore the charts, the grounded brief, and the review queue.

**With Docker** (proves it runs the same way on any machine — see `docs/DESIGN.md` §5.7 for why this was worth building):
```bash
docker build -t pulseai .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-your-key-here pulseai
```
Then visit `http://localhost:8501`.

**Run the test suite:**
```bash
PYTHONPATH=src python -m pytest tests/ -v
```

**Run the golden-set accuracy evaluation** (requires your real API key; prints category/sentiment/urgency accuracy, a confusion matrix, and a confidence-calibration table):
```bash
PYTHONPATH=src python -m pytest tests/golden/test_accuracy.py -v -s
```

**Run the zero-shot vs few-shot probe** (shows concretely what the few-shot examples fix):
```bash
PYTHONPATH=src python -m pulseai.prompts.zero_shot_probe
```

## Screenshots

_Add screenshots here before your demo:_
1. Run `streamlit run dashboard/app.py`, run an analysis, and take screenshots of: the full dashboard header/metrics, the Grounded Action Brief with an insight expanded to show its cited quotes, the category/sentiment/urgency charts, and the review queue tab.
2. Save them into `docs/screenshots/` (e.g. `dashboard-overview.png`, `grounded-brief.png`, `review-queue.png`).
3. Reference them here:
   ```markdown
   ![Dashboard overview](docs/screenshots/dashboard-overview.png)
   ![Grounded brief with citations](docs/screenshots/grounded-brief.png)
   ```

## The AI pipeline

```
Raw feedback
   │
   ▼
[1] Ingestion        → load & normalize from CSV
   ▼
[2] Validation       → empty/too-long/duplicate handling, all deterministic
   ▼
[3] Cleaning         → conservative text normalization, language detection
   ▼
[4] Per-item Analysis (LLM) → category + sentiment + urgency + themes, one combined call
   ▼
[5] Confidence Gate  → low-confidence items flagged for human review
   ▼
[6] Theme Clustering (embeddings, in-memory, no vector DB) → LLM names each cluster
   ▼
[7] Aggregation (pure code — never the LLM) → counts, distributions, priority scores
   ▼
[8] Weekly Summary (LLM, grounded + citation-validated) → the executive brief
   ▼
[9] Dashboard
```

Full stage-by-stage rationale, failure modes, and confidence handling: `docs/DESIGN.md` §6.

## Prompt strategy

- **Temperature 0 / low reasoning effort + a pinned model version** — the same input reliably produces the same output. Verified directly in `tests/test_analyze.py::test_consistency_same_input_same_scripted_output_same_result` and `tests/test_aggregate.py::test_aggregation_is_deterministic_across_repeated_calls`.
- **Few-shot examples are deliberately chosen, not random.** Each of the 13 examples in `analysis_v2` targets a specific boundary or failure mode (sarcasm, negation, mixed sentiment, category boundaries) — see the `_targets` field on every example in `src/pulseai/prompts/fewshot_analysis_v2.json` for the exact rationale.
- **Prompts are versioned.** `analysis_v1` → `analysis_v2` happened after the golden-set evaluation revealed three specific confusions (billing/praise, performance/bug, neutral/praise); v2 added three targeted examples to fix them. Because `prompt_version` is part of the SQLite cache key, bumping it automatically invalidates only the stale results — nothing needed manual cache-clearing.
- **Structured output contract + retry + fallback.** Every LLM call that returns JSON is validated against a Pydantic schema; a malformed response gets one retry, then a safe, clearly-flagged fallback — never a crash. See `core/analyze.py`, `core/themes.py`, `core/summarize.py`.

Full reasoning: `docs/DESIGN.md` §9 and §10.

## Key design decisions

- **No vector database.** Theme clustering uses embeddings + in-memory `AgglomerativeClustering` (deterministic — no random seed). A vector DB (Chroma/Pinecone/Qdrant) solves a scaling problem this project doesn't have at a few-thousand-item scale. See §5.2.
- **No RAG.** Classification/sentiment/urgency are per-item transformations, not retrieval-over-a-knowledge-base — there's nothing to retrieve.
- **Deterministic clustering algorithm chosen specifically for consistency** — `AgglomerativeClustering` has no random initialization, unlike k-means, so the same embeddings always produce the same clusters.
- **Aggregation is 100% code, never the LLM.** Counting and averaging are things a model does unreliably; `core/aggregate.py` is pure Python/collections logic.
- **SQLite with content-hash + prompt-version keyed caching.** Re-running the pipeline never re-analyzes or re-pays for an already-seen item under the same prompt version.

Full comparison tables and tradeoffs for every technology choice: `docs/DESIGN.md` §5.

## Testing

58 automated tests across ingestion, validation, cleaning, per-item analysis (including retry/fallback/consistency), storage/caching, theme clustering, aggregation, the grounded summary's citation-validation, and full end-to-end pipeline orchestration — run offline with fake clients, no API key required.

Separately, `tests/golden/` holds a 24-item hand-labeled golden dataset and a live evaluation harness (`run_evaluation.py`, wrapped by `test_accuracy.py`) that measures real classification accuracy, a confusion matrix, and confidence calibration against the actual model — this one requires a live API key and is what caught the v1→v2 prompt improvement documented above.

```bash
PYTHONPATH=src python -m pytest tests/ -v                            # 58 passed, 1 skipped (no key)
PYTHONPATH=src python -m pytest tests/golden/test_accuracy.py -v -s   # live accuracy report
```

## Known limitations

- **`langdetect` occasionally misfires on gibberish/very short input** (e.g. it once tagged pure keyboard-mashing as Somali rather than "unknown"). This doesn't break anything downstream — such items still get caught by the confidence gate — but it's a real, observed limitation worth knowing about.
- **The golden dataset is a 24-item seed, not a comprehensive benchmark.** DESIGN.md §17 recommends growing this to 40-60 items; the current set is enough to catch real confusions (and did) but a larger set would give more statistically reliable accuracy numbers.
- **LLM confidence is not perfectly calibrated.** The calibration table in the golden-set report shows this directly rather than assuming it — `DEFAULT_CONFIDENCE_THRESHOLD` in `core/analyze.py` is currently a placeholder and should be tuned from that calibration data, not guessed.
- **Theme clustering's `distance_threshold` (0.35) is an empirical starting point**, not a value tuned against a large, diverse corpus — it may need adjusting as real data volume grows.
- **The dashboard's "Run weekly analysis" button re-runs theme labeling and summary generation fresh every time** (only per-item analysis results are cached) — clicking it twice on identical data can produce a slightly different (though equally grounded) narrative each time, since those two stages aren't cached by content hash the way per-item analysis is.

## Future improvements

- Postgres for multi-user/concurrent access (SQLite's single-writer model is the right choice for this scale, not for a real multi-user product)
- A React frontend against the optional FastAPI layer, if this became a real product rather than a demo
- The "Emerging Issue Radar" nice-to-have (week-over-week theme velocity, requires 2+ weeks of data to be meaningful)
- Concurrent per-item analysis calls (10-20 in parallel) to cut batch runtime, once correctness is fully proven sequentially
- Caching theme-labeling and summary-generation calls by input hash, the same way per-item analysis is cached
- A larger, more diverse golden dataset (40-60+ items) for more statistically reliable accuracy tracking across prompt versions

## Lessons learned

> **This section is intentionally left as a template.** The rubric specifically grades your own honest reflection here (what was hardest, what you'd do differently, what you're least confident about) — filling it in for you would defeat the purpose of that grading criterion. Answer these for real, based on what actually happened while you built this:

**What took the longest? What broke that surprised you?**
_(Your answer — e.g. was it the theme clustering distance threshold, the confidence gate calibration, sorting out a few file-editing mishaps along the way, something else?)_

**If you started today, what would you change in your approach?**
_(Your answer)_

**What aspect of this are you least confident about?**
_(Your answer — LLM confidence calibration is a defensible honest answer here, but only if it's true for you)_

**What did you figure out that wasn't in the original brief?**
_(Your answer — e.g. the `langdetect` misfire on gibberish, or the decision to keep theme-labeling/summary calls uncached)_

## Roadmap status

- [x] Phase 1 — Project initialization
- [x] Phase 2 — Schemas & folder structure
- [x] Phase 3 — Ingestion, validation, cleaning
- [x] Phase 4 — Prompt engineering (analysis)
- [x] Phase 5 — Per-item analysis (LLM)
- [x] Phase 6 — Storage & caching
- [x] Phase 7 — Theme extraction
- [x] Phase 8 — Aggregation
- [x] Phase 9 — Weekly summary (hero feature)
- [x] Phase 10 — Dashboard
- [x] Phase 11 — Testing & evaluation
- [x] Phase 12 — Documentation & demo prep
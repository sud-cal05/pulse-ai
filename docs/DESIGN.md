# PulseAI — Customer Feedback Intelligence Platform
### Engineering Design Document (Architecture-First, No Code)

> **Status:** Design / pre-implementation
> **Author:** (you)
> **Audience:** Your mentor, senior reviewers, and future-you during the 2-week build
> **Rule for this doc:** Every decision names the rubric items it moves. Every technology that appears has to justify its existence.

---

## Table of Contents

1. [How to use this document](#1-how-to-use-this-document)
2. [Rubric decoding — where the points actually are](#2-rubric-decoding--where-the-points-actually-are)
3. [The hero feature (project personality)](#3-the-hero-feature-project-personality)
4. [System architecture overview](#4-system-architecture-overview)
5. [Major technology decisions](#5-major-technology-decisions)
   - [5.1 AI / LLM provider](#51-ai--llm-provider)
   - [5.2 Do we need RAG or a vector database?](#52-do-we-need-rag-or-a-vector-database)
   - [5.3 Backend / application layer](#53-backend--application-layer)
   - [5.4 Storage](#54-storage)
   - [5.5 Dashboard framework](#55-dashboard-framework)
   - [5.6 Charting library](#56-charting-library)
   - [5.7 Deployment](#57-deployment)
6. [The AI pipeline, stage by stage](#6-the-ai-pipeline-stage-by-stage)
7. [Classification taxonomy](#7-classification-taxonomy)
8. [JSON schemas (design, not prompts)](#8-json-schemas-design-not-prompts)
9. [Prompt design strategy](#9-prompt-design-strategy)
10. [Few-shot strategy](#10-few-shot-strategy)
11. [Per-task method decisions (sentiment / urgency / themes / summary)](#11-per-task-method-decisions)
12. [Folder structure](#12-folder-structure)
13. [Git strategy](#13-git-strategy)
14. [README plan](#14-readme-plan)
15. [Dashboard plan](#15-dashboard-plan)
16. [Testing plan](#16-testing-plan)
17. [Accuracy evaluation](#17-accuracy-evaluation)
18. [Failure modes & mitigations](#18-failure-modes--mitigations)
19. [Demo preparation (predicted questions + ideal answers)](#19-demo-preparation)
20. [Scope control — required vs nice-to-have](#20-scope-control)
21. [Implementation roadmap (12 phases)](#21-implementation-roadmap)
22. [Glossary](#22-glossary)

---

## 1. How to use this document

This is a plan, not code. Read sections 2–5 first: they decide the *shape* of everything else. Sections 6–11 decide the *AI behaviour*. Sections 12–20 decide *how the work is judged and delivered*. Section 21 is the day-by-day build order.

Two principles run through the whole thing:

- **The AI quality dimensions are worth the most.** Mission-Specific (30%) + AI Understanding (21%) = 51% of your grade. Spend your two weeks accordingly: the dashboard is a *delivery vehicle* for AI quality, not the main event. Do not burn a week on a React frontend.
- **Determinism and honesty about uncertainty are your two superpowers.** Half the rubric quietly rewards "same input → same output" and "you know where it fails." Design for both from day one rather than bolting them on.

---

## 2. Rubric decoding — where the points actually are

Before any architecture, here's what the rubric is *really* testing, grouped by the engineering behaviour it rewards. Read this as your north star.

| Rubric cluster | Weight | What it's really testing | Design response |
|---|---|---|---|
| **M5S Mission-Specific** | 30% | Blind-input accuracy, score consistency, deliberate few-shot, specific themes, actionable summary, readable charts | Golden dataset, temp 0, boundary-case few-shot, embeddings-backed themes, grounded summary |
| **M5A AI Understanding** | 21% | Can you *explain* the AI in plain English and know where it breaks | This whole document is your study guide; §19 is the drill |
| **M5B Integration Quality** | 18% | Consistency, edge cases, API-failure resilience, usable format, no secrets | Pydantic validation, retry + fallback, `.env`, confidence-gated review queue |
| **M5C Problem–Solution Fit** | 14% | Solves a real CX problem; a non-technical user can drive it end-to-end | Streamlit dashboard, real feedback data, complete flow |
| **M5D Learning Demonstrated** | 11% | Honest reflection, 2-week commit spread, "what I'd do differently" | §13 git cadence, §19 reflection answers |
| **M5E Code Quality** | 7% | Readable, runnable-from-README, no obvious security holes | §12 structure, §14 README, input validation |

**Key insight:** the two consistency items (M5B1, M5S2) and the classification-accuracy item (M5S1) are worth serious points and are the *easiest to lose accidentally*. A single non-zero temperature or an unpinned model can drop you two whole grades. Determinism is the highest-leverage decision in the project — it is baked into §9.

---

## 3. The hero feature (project personality)

Every project on the mentor's desk will classify feedback and draw a bar chart. You need one thing that makes them remember *yours*. I'm recommending one primary and giving two alternatives so you can choose based on your taste.

### Recommended: **The Grounded Action Brief** — an executive summary that cites its sources

The weekly summary is not free-form prose. Every insight it states is **attributed to the specific feedback IDs that support it**, ranked by a transparent **Priority Score**, and tagged with a **confidence band**. In the dashboard, each insight links to the exact customer quotes underneath it.

Why this is the right hero feature:

- **It's an anti-hallucination feature, by design.** The single biggest failure of an LLM summary is inventing a trend that isn't in the data. By forcing the model to cite feedback IDs and validating that those IDs exist, the summary *cannot* claim something the data doesn't support. This directly answers the scariest mentor question — "where does your AI lie to me?" — with "it structurally can't, and here's why."
- **It's genuinely production-grade.** Grounding/attribution is exactly how Intercom, Notion AI, and Anthropic's own products keep AI summaries trustworthy. It reads as senior, not student.
- **It's memorable in one sentence:** *"My executive summary cites its sources like a research paper, and ranks issues by business impact."*

**Priority Score** (transparent, explainable — not a black box):
```
priority_score = volume_share × avg_urgency_weight × negativity_weight
```
where each factor is normalised 0–1. Executives get a *ranked* action list, not a wall of themes.

**Rubric impact:** M5S5 (actionable summary — huge), M5A4 (knows where AI fails — you engineered against it), M5S4 (specific themes — grounding forces specificity), M5B4 (usable output), M5C1 (solves the real problem).

### Alternative A: **Emerging Issue Radar**
Detect themes whose week-over-week frequency is *accelerating* (z-score on theme velocity) and surface them before they become top complaints. Strong "early warning" story; slightly more data plumbing (needs ≥2 weeks of data to shine, which is awkward in a demo).

### Alternative B: **Confidence-Gated Triage Queue**
Every item the AI is unsure about is auto-routed to a "needs human review" queue. Honest and elegant, but on its own it's a smaller "wow" than the grounded brief. **Recommendation: build this as a *supporting mechanic* of the main system regardless** — it's cheap and it wins M5B2 + M5A4.

> **Final call:** Grounded Action Brief as the hero, with the confidence-gate baked in everywhere as connective tissue. Emerging Issue Radar becomes a nice-to-have if you have time in week 2.

---

## 4. System architecture overview

The cleanest architecture for a solo 2-week build is a **decoupled core with thin delivery layers**:

```
                    ┌──────────────────────────────┐
                    │   Streamlit Dashboard (UI)    │  ← thin, imports core
                    └───────────────┬──────────────┘
                                    │
        (optional nice-to-have)     │
   ┌────────────────┐               │
   │  FastAPI layer  │──────────────┤   ← both are thin clients of the core
   └────────────────┘               │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │            CORE PIPELINE (pure Python)         │
        │  ingest → validate → clean → analyse (LLM) →   │
        │  cluster themes → aggregate → summarise (LLM)  │
        └───────────────┬───────────────────────────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
      ┌──────────────┐   ┌──────────────────┐
      │  SQLite DB    │   │  LLM provider API │
      │ (results +    │   │  (temp 0, pinned  │
      │  raw JSON)    │   │  model version)   │
      └──────────────┘   └──────────────────┘
```

**The one rule that matters:** the core pipeline knows nothing about Streamlit, FastAPI, or SQLite specifics. It takes feedback in, returns validated objects out. The UI and the API are *thin clients*. This single decision earns M5E1 (readable structure), M5B (each stage is independently testable), and lets you say in the demo: *"the AI logic is completely decoupled from the interface — I could put a Slack bot on it tomorrow."*

**Rubric impact:** M5E1, M5C3 (right interface for analytics = dashboard), M5B (testable integration), M5D5 (thoughtful design beyond the brief).

---

## 5. Major technology decisions

For each: alternatives → comparison → recommendation → why it fits *this* project → risks → future scalability.

### 5.1 AI / LLM provider

**Alternatives considered:** hosted frontier API (OpenAI / Anthropic Claude / Google Gemini) vs a local open-weights model (e.g., Llama/Mistral via Ollama).

| Option | Pros | Cons | Industry usage |
|---|---|---|---|
| **Hosted API (OpenAI / Claude / Gemini)** | Strongest accuracy; native structured-output / tool-use for guaranteed JSON; zero infra; fast to build | Costs money; network dependency; rate limits | Default for virtually every AI product startup |
| **Local LLM (Ollama etc.)** | Free per-call; offline; data privacy | Weaker accuracy on nuanced classification; you manage GPU/infra; slower on a laptop; JSON adherence is shakier | Regulated/privacy-sensitive or high-volume cost-sensitive shops |

**Recommendation:** a **hosted frontier API with native structured-output support** (OpenAI Structured Outputs *or* Anthropic Claude tool-use — either is fine; pick one and pin the model version). Wrap it behind your own `LLMClient` interface so the provider is swappable in one file.

**Why for this project:** the rubric grades *accuracy* and *JSON adherence* directly (M5S1, M5B4). Native structured outputs give you schema-guaranteed JSON, which kills a whole class of parsing bugs and directly wins the "usable format" and "consistency" items. A local model on an internship laptop would cost you accuracy points to save money you don't need to save at this scale (a few thousand feedback items is cents).

**Risks:** vendor rate limits / key expiry → mitigated by the retry + graceful-fallback design (§18) and by never committing the key (M5B5). Cost → cap it: run the full corpus once, cache results in SQLite, and *never re-analyse an item you've already analysed* (idempotency by content hash).

**Future scalability:** the `LLMClient` abstraction means you can swap to a local model later for cost, or route cheap items to a small model and hard items to a big one, without touching the pipeline.

**Rubric impact:** M5S1, M5B4, M5B1, M5A2 (you can explain *why* hosted-with-structured-output).

---

### 5.2 Do we need RAG or a vector database?

**Short answer: No RAG. No heavyweight vector DB. But yes to embeddings for one narrow job.**

This is the most important "engineering judgment" decision in the project, and the rubric will probe it (M5A2 literally asks *"why ChromaDB and not keyword search?"*). Getting this right — by *declining* unnecessary tech — is worth more than adding it.

**What RAG actually is:** retrieval-augmented generation feeds an LLM relevant *retrieved documents* so it can answer questions grounded in a knowledge base. It solves "answer questions over documents the model wasn't trained on."

**Why your project doesn't need it:** classification, sentiment, and urgency are *per-item transformations* — the model reads one piece of feedback and labels it. There is nothing to *retrieve*. Forcing RAG here would be résumé-driven design, and a good mentor will catch it.

**Where embeddings *do* earn their place — corpus-level theme extraction and dedup:**

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **No embeddings; LLM names themes from raw batches** | Simplest | Struggles to cluster consistently across large corpora; "theme explosion" | Fallback only |
| **Embeddings + clustering + LLM labelling** | Groups semantically similar feedback reliably → *specific* themes; enables dedup | One extra dependency | ✅ Recommended |
| **Full vector DB (Chroma/Qdrant/Pinecone)** | Persistent similarity search, scales to millions | Massive overkill for a few thousand items that fit in RAM | ❌ Overengineering |

**Recommendation:** compute embeddings for feedback, cluster them **in memory** (a few thousand vectors is trivial — use cosine similarity with `scikit-learn` / `numpy`, or `HDBSCAN` if you want automatic cluster counts), then have the LLM *name* each cluster with a specific label + representative quotes. **No Chroma, no Pinecone** — the data volume doesn't justify a database whose entire purpose is scaling similarity search beyond memory.

**The comparison you'll be asked about (memorise this):**
- *Chroma / Qdrant* — local-first vector DBs, great when you need persistent semantic search at scale. Overkill here.
- *Pinecone* — managed, production, millions of vectors. Way overkill.
- *FAISS* — a library (not a DB) for fast in-memory similarity. Closest to appropriate, but even FAISS is unnecessary for a few thousand vectors — plain cosine similarity is enough.

**Why this wins:** you can say *"I chose embeddings for theme clustering because keyword matching can't tell that 'app keeps crashing' and 'constant force-closes' are the same issue. I did **not** add a vector database because my corpus fits in memory — a vector DB solves a scaling problem I don't have."* That single answer nails M5A2 and M5D5 (research beyond the brief, and the maturity to *not* add tech).

**Rubric impact:** M5A2 (the flagship question), M5S4 (specific themes), M5B2 (dedup handles duplicate feedback edge case), M5D4/M5D5.

---

### 5.3 Backend / application layer

**Alternatives:** FastAPI · Flask · Django · Express (Node).

| Option | Pros | Cons | Industry usage |
|---|---|---|---|
| **FastAPI** | Async; **Pydantic built in** (schema validation = free rubric points); auto OpenAPI docs; modern | Slightly more concepts than Flask | Stripe-adjacent modern Python APIs, ML services |
| **Flask** | Dead simple, tiny | No built-in validation; you'd add Pydantic anyway | Legacy / tiny services |
| **Django** | Batteries included, ORM, admin | Heavy; built for CRUD web apps, not AI microservices | Content-heavy web apps |
| **Express (Node)** | Fine if your stack were JS | Splits you off Python, away from the ML ecosystem | JS shops |

**Recommendation:** **FastAPI — but only as an optional thin layer.** Your *required* deliverable is the pure-Python core + Streamlit. FastAPI is a **nice-to-have** that (a) shows API-design skill and (b) gives Pydantic a natural home. Even if you skip the HTTP endpoints, **use Pydantic models for every schema** — that's where the real value is.

**Why for this project:** Pydantic models *are* your JSON schemas (§8). They validate LLM output, reject malformed responses, and generate docs for free. That single library moves M5B4 (usable format), M5E3 (input validation / security), and M5A (you can point at typed schemas and explain them).

**Risks:** scope creep — don't build a full REST API and neglect AI quality. Keep FastAPI to 1–2 endpoints or skip it.

**Future scalability:** if this became real SaaS, the FastAPI layer is where auth, multi-tenancy, and webhooks would live. Note that in §20 as "future work."

**Rubric impact:** M5B4, M5E3, M5A, M5D5.

---

### 5.4 Storage

**Alternatives:** SQLite · PostgreSQL · MongoDB · flat files (CSV/JSON).

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **SQLite** | Zero-config, file-based, ships in the repo, perfect for demos, full SQL | Single-writer, not for high concurrency | ✅ Recommended |
| **Postgres** | Production-grade, concurrent | Needs a server; overkill for one user | Future scaling |
| **MongoDB** | Flexible JSON docs | You'd lose SQL aggregation ease; your data is actually well-structured | ❌ |
| **Flat CSV/JSON** | Trivially simple | No querying, no integrity, messy for aggregation | Sample data only |

**Recommendation:** **SQLite**, with a thin data-access module so it can swap to Postgres later. Store each feedback row *and* the full validated analysis JSON in a column (hybrid relational + JSON). Key the table on a **content hash** so re-runs are idempotent (never pay to analyse the same text twice — this also guarantees M5B1/M5S2 consistency on re-runs, because you serve the cached result).

**Why for this project:** the mentor has to *run* your project (M5E2). "Clone, `pip install`, run — no database server to set up" is a massive win for reproducibility. And SQL `GROUP BY` makes your aggregation stage (§6) trivial and *deterministic* (code, not LLM).

**Risks:** none meaningful at this scale.

**Future scalability:** the data-access layer is the swap point to Postgres for multi-user. Document it.

**Rubric impact:** M5E2 (runs from README), M5B1 (cached = consistent), M5C4 (complete, works end-to-end).

---

### 5.5 Dashboard framework

**Alternatives:** Streamlit · React (SPA) · Next.js · static/exported charts.

| Option | Pros | Cons | Industry usage |
|---|---|---|---|
| **Streamlit** | Python-native (no context switch); a polished analytics dashboard in hours; free hosting | Less pixel-perfect control; not for consumer apps | Internal ML/data tools everywhere |
| **React SPA** | Full control, production-grade UI | Eats a week you don't have; splits your stack | Customer-facing products |
| **Next.js** | React + SSR + routing | Even more setup; total overkill here | Marketing sites, full products |
| **Static charts** | Trivial | Not interactive, not "a tool" | Reports only |

**Recommendation:** **Streamlit.** This is a deliberate trade: you're spending your saved frontend time on AI quality, which is worth ~3× more in the rubric.

**Why for this project:** M5C2 requires a *non-technical user to operate it unassisted* — Streamlit's clean, guided layout is ideal. M5C3 rewards the *right interface for analytics* — a dashboard, which Streamlit does natively. You stay in Python, so the whole project is one language and one mental model (helps M5E1 readability).

**Risks:** Streamlit can look samey. Mitigate with deliberate layout, clear titles, and Plotly charts (§5.6). The hero feature (grounded brief with clickable quotes) is what makes it memorable, not the framework.

**Future scalability:** if it became a product, you'd rebuild the UI in React against the FastAPI layer. Note as future work — this shows you know Streamlit's ceiling (good for M5D4).

**Rubric impact:** M5C2, M5C3, M5S6 (readable charts), and *indirectly* the whole AI cluster by freeing your time.

---

### 5.6 Charting library

Your chart library follows from the dashboard framework. Since you're on Streamlit (Python), the JS libraries in the brief (Chart.js/Recharts/ECharts) don't apply directly — those are for a React path. Here's the honest mapping:

| Option | Fits | Pros | Cons |
|---|---|---|---|
| **Plotly** (via Streamlit) | ✅ Streamlit | Interactive, hover tooltips, clean labels, export to PNG | Slightly heavier |
| **Altair** (via Streamlit) | ✅ Streamlit | Elegant declarative grammar, great defaults | Less interactive |
| Streamlit native charts | ✅ Streamlit | One-liners | Limited customisation |
| Chart.js / Recharts / ECharts | React only | (Recharts) clean React charts; (ECharts) most powerful | Require the React path you're not taking |

**Recommendation:** **Plotly** for the main charts (interactive, self-labelling tooltips → directly wins M5S6 "interpretable without explanation"), Altair or native for simple ones.

**Why:** M5S6 explicitly checks "could a stakeholder understand the charts without you explaining them?" Plotly's hover labels + clear axis titles make charts self-documenting. Always label axes, add units, and title every chart with the *question it answers* ("Which categories are customers complaining about most?").

**Rubric impact:** M5S6, M5C3.

---

### 5.7 Deployment

**Alternatives:** local-only · Docker · Railway · Render · Streamlit Community Cloud · Vercel.

| Option | Pros | Cons | Fit |
|---|---|---|---|
| **Docker** | Reproducible anywhere; "works on my machine" solved | Small learning curve | ✅ Required-ish |
| **Streamlit Community Cloud** | Free live URL for a Streamlit app in minutes | Streamlit-only | ✅ Nice-to-have live link |
| **Railway / Render** | Simple PaaS for the FastAPI layer | Only if you build the API | If API exists |
| **Vercel** | Great for Next.js/JS frontends | You're not on JS | ❌ |
| **Local only** | Zero effort | Reproducibility risk for the mentor | Baseline |

**Recommendation:** **Dockerfile (required)** for reproducibility + **Streamlit Community Cloud (nice-to-have)** for a live demo link. A `Dockerfile` + one-command run is the single strongest thing you can do for M5E2 ("another developer can run it").

**Why:** the reviewer literally tries to run your project from the README. `docker build && docker run` sidesteps every environment mismatch. A live Streamlit URL means the mentor can click a link during the demo — high polish, low effort.

**Risks:** don't put your API key in the image. Pass it as an env var at runtime (`docker run -e ...` or Streamlit Cloud secrets). This is M5B5 (no hardcoded secrets).

**Rubric impact:** M5E2, M5B5, M5C4.

---

### Technology decisions — summary table

| Layer | Choice | Status | Primary rubric wins |
|---|---|---|---|
| LLM | Hosted frontier API + structured output, pinned version, temp 0 | Required | M5S1, M5B1, M5B4 |
| Retrieval | Embeddings for theme clustering; **no** vector DB | Required | M5A2, M5S4 |
| Validation | Pydantic schemas | Required | M5B4, M5E3 |
| Backend | Pure-Python core; FastAPI thin layer | Core required / API optional | M5E1, M5D5 |
| Storage | SQLite (content-hash keyed, JSON column) | Required | M5E2, M5B1 |
| Dashboard | Streamlit | Required | M5C2, M5C3 |
| Charts | Plotly | Required | M5S6 |
| Deploy | Dockerfile + Streamlit Cloud | Docker required / cloud optional | M5E2, M5B5 |

---

## 6. The AI pipeline, stage by stage

```
Raw feedback
   │
   ▼
[1] Ingestion        → load from CSV/JSON/DB into a normalised record
   │
   ▼
[2] Validation       → reject/flag malformed, empty, duplicate records
   │
   ▼
[3] Cleaning         → normalise whitespace, strip signatures, detect language
   │
   ▼
[4] Per-item Analysis (LLM, ONE structured call) → category + sentiment + urgency + item themes + confidences + key quote
   │
   ▼
[5] Confidence Gate  → low-confidence items flagged needs_human_review
   │
   ▼
[6] Theme Clustering (embeddings) → group items into corpus-level themes, LLM names each
   │
   ▼
[7] Aggregation (code, deterministic) → counts, distributions, priority scores
   │
   ▼
[8] Weekly Insights (LLM, ONE grounded call) → executive brief with cited feedback IDs
   │
   ▼
[9] Persistence + Dashboard payload
```

**Design decision — one combined analysis call vs one call per task.**

| Approach | Pros | Cons |
|---|---|---|
| **Combined per-item call** (category+sentiment+urgency+themes together) | Fewer API calls (cheaper/faster); atomic; one schema; naturally consistent | One schema to get right; a parse failure loses all four labels for that item |
| Separate call per task | Modular; tune each independently | 4× cost & latency; 4× parse-failure surface; harder to keep consistent |

**Recommendation: combined per-item call.** For a few-field labelling task the model handles all four dimensions well in one pass, and it's dramatically cheaper and more consistent. Keep the *aggregation* and *summary* as separate stages because they operate on the whole corpus, not one item.

### Stage-by-stage rationale

**[1] Ingestion** — *Why it exists:* decouples data source from logic; normalises heterogeneous inputs (CSV export, API dump) into one `FeedbackRecord`. *Failure case:* wrong encoding / weird delimiters → handle with explicit parsing and a clear error, not a crash (M5B3).

**[2] Validation** — *Why:* garbage in, garbage out; also the "edge cases" rubric item lives here (M5B2). Empty string, whitespace-only, absurdly long, duplicate (by content hash). *Behaviour:* empty → skip with a logged reason, don't send to the LLM; very long → truncate to a token budget with a flag; duplicate → reuse cached result. *Confidence:* n/a (deterministic).

**[3] Cleaning** — *Why:* removes noise (email signatures, quoted replies, excessive whitespace) that skews the model; detects language so non-English can be flagged/handled (M5B2 "wrong language"). *Failure case:* over-aggressive cleaning strips meaning → keep cleaning conservative; store both raw and cleaned text.

**[4] Per-item Analysis (LLM)** — *Why:* the core transformation. Returns the structured `PerItemAnalysis` object. *Failure cases:* hallucinated category, wrong sentiment on sarcasm/negation, over-confident. *Confidence:* model self-reports a 0–1 confidence per field — treat as a **coarse triage signal, not truth** (LLM confidence is poorly calibrated; validate against the golden set). *Determinism:* temperature 0 + pinned model + structured output.

**[5] Confidence Gate** — *Why:* turns "the AI is sometimes wrong" into a *designed* behaviour. Items below a threshold → `needs_human_review = true` and surfaced in a review queue. This is your honest answer to M5A4. *Threshold:* pick empirically from the golden set (e.g., where accuracy drops below X).

**[6] Theme Clustering** — *Why:* per-item themes are noisy; corpus-level themes need semantic grouping (§5.2). Embed → cluster → LLM labels each cluster with a *specific* name + representative quotes. *Failure case:* "theme explosion" (too many tiny themes) → cap cluster count / merge small clusters / use min-cluster-size. *Confidence:* cluster cohesion (avg intra-cluster similarity).

**[7] Aggregation (code)** — *Why:* all distributions, counts, and Priority Scores are computed in **deterministic Python/SQL, not the LLM.** This is critical: never ask an LLM to count things — it's unreliable and non-deterministic. Doing the maths in code guarantees M5S2/M5B1 consistency and makes charts trustworthy. *Failure case:* none if it's pure code.

**[8] Weekly Insights (LLM)** — *Why:* turns the deterministic aggregate into an actionable, **grounded** narrative (the hero feature). Input = the aggregate + top themes + representative quotes *with their IDs*; output = insights each citing feedback IDs. *Failure case:* hallucinated trend → mitigated by (a) feeding only real aggregated data, (b) requiring ID citations, (c) validating cited IDs exist and dropping any insight that cites a non-existent ID. *Confidence:* per-insight confidence + a global caveat line.

**[9] Persistence + payload** — *Why:* cache everything by content hash for idempotency and speed; assemble the denormalised `DashboardPayload` the UI reads. *Failure case:* stale cache after a prompt change → include `prompt_version` in the cache key so a new prompt version re-analyses.

**Rubric impact across the pipeline:** M5A3 (you can walk through input→output), M5A4 (failure points named per stage), M5B1/M5S2 (determinism designed in), M5B2 (validation stage), M5S4 (clustering), M5S5 (grounded summary).

---

## 7. Classification taxonomy

A good taxonomy is **mutually exclusive enough to be consistent, and exhaustive enough that "Other" stays small.** Too many categories → the model can't decide (hurts M5S1/M5S2). Too few → themes get mushy. Aim for ~10–12.

| Category | Definition | Why it exists | Edge / ambiguous cases |
|---|---|---|---|
| **Bug / Defect** | Something broken or behaving wrong | The highest-action category for eng | Perf issues (route to Performance); "feature doesn't exist" → Feature Request, not Bug |
| **Performance** | Slowness, timeouts, crashes, latency | Distinct engineering owner; different fix path than a logic bug | "App crashes" = Performance or Bug? Rule: crashes/hangs/slowness → Performance |
| **Feature Request** | Asking for new capability | Product roadmap signal | "It should do X" (missing) vs "X is broken" (Bug) |
| **UI / UX** | Confusing, ugly, hard-to-use interface | Design owner; not a functional bug | "Can't find the button" = UI/UX; "button does nothing" = Bug |
| **Authentication / Access** | Login, SSO, permissions, lockouts | High-urgency, security-adjacent | Password reset (Auth) vs account deletion (Account Mgmt/Other) |
| **Billing / Pricing** | Charges, invoices, refunds, plan cost | Revenue-critical, distinct CX team | "Too expensive" (Pricing sentiment) vs "charged twice" (Billing error) |
| **Documentation** | Missing/unclear docs, guides, API refs | Cheap, high-impact fixes | "I didn't know it could do X" — docs gap vs discoverability (UI/UX) |
| **Customer Support** | Experience with the support team itself | Meta-feedback about CX quality | Praise/complaint about an agent, not the product |
| **Security / Privacy** | Data exposure, vulnerabilities, trust | Critical urgency; must never be missed | Always high urgency; overlaps Auth — keep separate for visibility |
| **Integrations** | Third-party connections, APIs, webhooks | Common B2B pain; distinct owner | API errors (Integrations) vs internal API bug (Bug) |
| **Praise / Positive** | General positive sentiment, no ask | Balances the picture; morale + what's working | Positive but with a request → primary = the request category |
| **Other** | Genuinely doesn't fit | Escape hatch to prevent forced mislabels | Keep it small; a large "Other" means the taxonomy needs a new category |

**Handling multi-category feedback:** allow a `primary_category` (single, for clean stats) plus optional `secondary_categories` (list). This resolves the "one message, two issues" problem without polluting the primary distribution. *Explain in the demo why single-primary keeps charts interpretable (M5S6) while secondaries preserve nuance.*

**Ambiguous input rule of thumb (put this in your system prompt):** classify by the *action the company would take*, not the words used. "The app is slow and the docs didn't help" → primary = Performance (the actionable problem), secondary = Documentation.

**Rubric impact:** M5S1 (accuracy — clean boundaries reduce errors), M5S3 (you can justify each category), M5A5 (each category maps to a business owner).

---

## 8. JSON schemas (design, not prompts)

These are your Pydantic models. Every field earns its place; I note *why* each exists. No prompts here — just contracts.

### 8.1 `FeedbackRecord` (input, post-ingestion)
```
feedback_id: str            # stable ID; content hash → idempotency & dedup
source: str                 # csv | api | manual — provenance for debugging
customer_id: str | None     # to spot "one loud customer" vs a real trend
raw_text: str               # original, never mutated (auditability)
cleaned_text: str           # what actually goes to the model
language: str               # ISO code; drives non-English handling
created_at: datetime        # for weekly bucketing & trend/velocity
```

### 8.2 `PerItemAnalysis` (LLM output, the core object)
```
feedback_id: str                    # links back to the record
primary_category: Category(enum)    # single → clean distribution charts
secondary_categories: list[Category]# nuance without polluting primary stats
category_confidence: float 0–1      # feeds the confidence gate
sentiment: Sentiment(enum)          # positive | neutral | negative | mixed
sentiment_score: float -1..1        # magnitude for trend lines & heat
sentiment_confidence: float 0–1
urgency: Urgency(enum)              # low | medium | high | critical
urgency_score: float 0–1            # ranking within a level
urgency_signals: list[str]          # WHY urgent ("mentions data loss") → explainable, anti-black-box
item_themes: list[str]              # short specific tags; raw material for clustering
key_quote: str                      # the most representative span → grounding & dashboard
needs_human_review: bool            # set by the confidence gate
model_version: str                  # reproducibility & cache invalidation
prompt_version: str                 # ditto; ties to §9 versioning
analyzed_at: datetime
```
*Why `urgency_signals`:* it makes urgency **explainable** instead of a mystery number — a direct answer to M5A3/M5A4 and a trust win. *Why confidences per field:* the confidence gate and calibration analysis (§17) depend on them.

### 8.3 `ThemeCluster` (corpus-level, from stage 6)
```
theme_id: str
label: str                  # SPECIFIC: "checkout timeout on mobile", not "issues"
description: str            # one sentence
member_feedback_ids: list[str]
size: int                   # volume → priority score
avg_sentiment_score: float
avg_urgency_score: float
representative_quotes: list[{feedback_id, text}]  # grounding for the brief
cohesion: float             # cluster tightness → theme confidence
priority_score: float       # volume × urgency × negativity, normalised
period: str                 # week bucket → enables velocity/Emerging Radar
```

### 8.4 `WeeklyAggregate` (deterministic, code-computed)
```
period_start, period_end: date
total_feedback: int
category_distribution: dict[Category, int]
sentiment_distribution: dict[Sentiment, int]
urgency_distribution: dict[Urgency, int]
top_themes: list[ThemeCluster]        # ranked by priority_score
review_queue_count: int               # how many items the AI wasn't sure about
```
*Why code-computed:* counts must be exact and identical on re-run (M5S2/M5B1). LLMs cannot be trusted to count.

### 8.5 `ExecutiveSummary` (the Grounded Action Brief — hero)
```
headline: str
key_insights: list[{
    statement: str,
    supporting_feedback_ids: list[str],   # MUST reference real IDs → validated
    theme_id: str | None,
    confidence: float 0–1,
    priority_score: float
}]
recommended_actions: list[{
    action: str,
    rationale: str,
    expected_impact: str,                 # e.g. "affects ~18% of negative feedback"
    linked_theme_id: str,
    priority: int
}]
watch_items: list[str]                     # emerging / low-volume-but-rising
caveats: str                               # honesty: sample size, low-confidence note
generated_at, model_version, prompt_version
```
*Why `supporting_feedback_ids` is the whole ballgame:* it's the anti-hallucination guarantee. Post-generation you validate every ID exists; drop any insight that cites a fake one. That validation step is your proof to the mentor.

### 8.6 `DashboardPayload` (denormalised, what the UI reads)
```
aggregate: WeeklyAggregate
summary: ExecutiveSummary
items: list[PerItemAnalysis]     # for the drill-down table + review queue
themes: list[ThemeCluster]
trend_series: {...}              # week-over-week arrays for line charts
generated_at
```
*Why a single payload object:* the UI stays dumb — it renders one well-typed object. Clean separation (M5E1), and the payload is trivially cacheable.

**Rubric impact:** M5B4 (structured, usable), M5E3 (validated), M5A (typed, explainable), M5S5 (grounded summary contract).

---

## 9. Prompt design strategy

Prompts are graded indirectly through accuracy (M5S1), consistency (M5S2/M5B1), and your ability to *explain your choices* (M5S3). Design principles before any wording:

**Responsibilities by layer:**
- **System prompt** — the model's *identity and rules*: role ("you are a CX feedback analyst"), the taxonomy with definitions, the ambiguity rule ("classify by the action the company would take"), the output contract, and hard constraints ("output only valid JSON matching the schema; never invent a category outside the enum"). This is the stable, versioned part.
- **Developer / few-shot layer** — the *deliberately chosen examples* (§10) that pin down boundary behaviour. Stable, versioned.
- **User prompt** — just the *data*: the cleaned feedback text (and for the summary stage, the aggregate + quotes). Nothing else.

**Output constraints:**
- **Structured outputs / tool-use with a strict JSON schema** — the model is *forced* to return schema-valid JSON. This is the single biggest reliability lever (M5B4).
- **Temperature 0** — near-deterministic. This is non-negotiable for M5B1/M5S2. Explain in the demo: *"temperature controls randomness; 0 means the model picks the most likely token every time, so the same input gives the same output."*
- **Pinned model version** — a silent model update would change your outputs. Pin it and record it in every result (`model_version`).

**Reliability scaffolding:**
- **Validate → retry once → fallback.** Parse with Pydantic. On failure, retry once (models occasionally slip). On second failure, return a safe default (`primary_category = "Other"`, `needs_human_review = true`) and log it — **never crash** (M5B3).
- **Guardrails.** Reject enum values outside the taxonomy at parse time. Validate cited feedback IDs in the summary. Cap input length.
- **Prompt versioning.** Store prompts as versioned files (`v1`, `v2`) and stamp `prompt_version` on every output. This lets you (a) invalidate cache correctly and (b) run regression tests when you change a prompt (§16). It's also a great "learning demonstrated" artifact — your git history shows prompt iteration (M5D1/M5D3).

**Why deterministic output matters (memorise for M5A):** the rubric runs identical input twice, 5 minutes apart, and compares. If your temperature is > 0 or your model isn't pinned, you can *fail this by luck*. Determinism + cache = you pass it every time.

**Decision — chain-of-thought?** Reasoning helps accuracy but *hurts* JSON cleanliness and determinism, and leaks tokens. **Recommendation:** avoid free-form CoT. If you want reasoning for hard urgency calls, capture it as a *structured* field (`urgency_signals`) rather than prose — you get the accuracy benefit *and* a clean schema *and* explainability. Best of all worlds.

**Rubric impact:** M5S1, M5S2, M5B1, M5B3, M5B4, M5S3, M5D1/M5D3.

---

## 10. Few-shot strategy

**How many examples?** Not "as many as possible." For this task, **~8–12 total, chosen for boundaries**, beats 30 random ones. Each example costs tokens on every call; the goal is *coverage of decision boundaries*, not volume.

**What to include (and why):**
- **Positive/clear examples (2–3):** anchor the format and obvious cases.
- **Boundary examples (the important ones, 4–6):** the exact confusions from §7 — "app slow *and* docs unhelpful" (primary vs secondary), "too expensive" (Pricing) vs "charged twice" (Billing), crash (Performance) vs logic error (Bug). These teach the *rule*, not just the pattern.
- **Hard-language examples (2–3):** sarcasm ("oh great, another crash, love it" → negative, not positive), negation ("not bad at all" → positive), mixed sentiment ("love the UI but billing is a nightmare" → mixed).
- **Edge examples (1–2):** emoji-only, very short, non-English → show the desired *graceful* behaviour.

**Why few-shot improves consistency (tie to rubric):** examples collapse the model's ambiguity about *your* taxonomy into a narrow, repeatable behaviour. Two similar inputs get labelled the same way because the boundary is demonstrated, not guessed. That's a direct win for **M5S1 (accuracy)** and **M5S2/M5B1 (consistency)**. And because each example is *deliberate*, you can answer **M5S3** ("why these examples?") with: *"each one pins a specific boundary I saw the model get wrong in zero-shot testing."* That sentence — grounded in your own testing — is a top-marks answer for M5S3 and M5D5.

**Process (do this, it's a rubric goldmine):** run zero-shot first on your golden set, *record the mistakes*, then add a few-shot example targeting each mistake class. Your commit history now literally shows "encountered a problem and solved it" (M5D5) and "consistent progress" (M5D3).

**Rubric impact:** M5S1, M5S2, M5S3, M5D5, M5D1.

---

## 11. Per-task method decisions

### Sentiment
| Option | Pros | Cons |
|---|---|---|
| **LLM prompt-based (in the combined call)** | Handles sarcasm/negation/context far better than lexicons; no extra infra; consistent with the rest | Costs a token or two more |
| Dedicated classifier model (HF) | Fast, cheap per call | Extra infra; weaker on sarcasm; another dependency |
| Lexicon (VADER etc.) | Trivial | Fails on sarcasm/negation/context — exactly your hard cases |
| Hybrid | Cross-check | Complexity not justified here |

**Recommendation:** **LLM-based, inside the combined per-item call.** Sarcasm and negation are explicit rubric failure modes (§18); only the LLM handles them well. Add a `mixed` label so genuinely two-sided feedback isn't forced positive/negative — this is a maturity signal.

### Urgency
**Recommendation:** LLM-based level (`low/medium/high/critical`) **plus a structured `urgency_signals` list** explaining *why*. Rule-anchor it in the system prompt (data loss / security / churn intent / outage = critical). The signals make it explainable and consistent. **Do not** compute urgency with keyword rules alone — "not urgent at all" would trip a naive keyword matcher.

### Theme extraction
**Recommendation (from §5.2):** **embeddings → in-memory clustering → LLM labelling.** Rejected alternatives: pure keyword/TF-IDF (misses paraphrase — "crashes" vs "force closes"), pure LLM-over-raw-batch (inconsistent, theme explosion), vector DB (overkill). The two-stage approach gives *specific* themes (M5S4) and representative quotes for the grounded brief.

### Weekly summary
| Option | Pros | Cons |
|---|---|---|
| Single prompt over all raw feedback | Simple | Blows the context window; unreliable; hallucination-prone; non-deterministic counts |
| **Multi-stage: deterministic aggregate → single grounded LLM summary** | Accurate counts (code); grounded, cited narrative (LLM); cheap; reliable | Two steps |

**Recommendation:** **multi-stage (map-reduce).** Compute all numbers in code, then hand the LLM a compact, factual aggregate + labelled quotes and ask *only* for the narrative + citations. This is the standard production pattern and it's what makes the hero feature trustworthy. Never dump 1,000 raw items into one summarisation prompt.

**Rubric impact:** M5S4, M5S5, M5A2, M5B1.

---

## 12. Folder structure

Each folder has exactly one responsibility. This structure scales because the *core* never imports the *delivery layers*.

```
pulseai/
├── README.md
├── Dockerfile
├── requirements.txt (or pyproject.toml)
├── .env.example              # documents required keys; NEVER commit real .env
├── .gitignore                # must include .env, *.db, __pycache__
│
├── src/pulseai/
│   ├── core/                 # PURE pipeline — no UI, no web framework
│   │   ├── ingest.py         # load & normalise raw feedback → FeedbackRecord
│   │   ├── validate.py       # empty / long / duplicate / language checks
│   │   ├── clean.py          # normalise text, strip signatures, detect language
│   │   ├── analyze.py        # combined per-item LLM call → PerItemAnalysis
│   │   ├── themes.py         # embeddings → cluster → LLM labelling
│   │   ├── aggregate.py      # deterministic counts + priority scores
│   │   ├── summarize.py      # grounded exec brief + ID validation
│   │   └── pipeline.py       # orchestrates stages 1→9
│   │
│   ├── ai/
│   │   ├── llm_client.py     # provider abstraction (swap in one place)
│   │   ├── embeddings.py     # embedding calls
│   │   └── config.py         # model version, temperature, thresholds
│   │
│   ├── prompts/              # versioned prompt text (v1/, v2/...)
│   │   ├── system_analysis_v1.txt
│   │   ├── fewshot_analysis_v1.json
│   │   └── system_summary_v1.txt
│   │
│   ├── schemas/              # Pydantic models = the JSON contracts (§8)
│   │   ├── feedback.py
│   │   ├── analysis.py
│   │   ├── themes.py
│   │   └── report.py
│   │
│   ├── storage/
│   │   ├── db.py             # SQLite access; content-hash idempotency
│   │   └── cache.py          # skip re-analysis of seen (text, prompt_version)
│   │
│   ├── api/                  # OPTIONAL FastAPI thin layer (nice-to-have)
│   │   └── main.py
│   │
│   └── utils/
│       ├── hashing.py        # content hash for IDs/dedup
│       └── logging.py
│
├── dashboard/
│   └── app.py                # Streamlit UI — reads DashboardPayload only
│
├── tests/
│   ├── test_validation.py
│   ├── test_consistency.py   # same input twice → equal (M5B1/M5S2)
│   ├── test_schema.py        # malformed LLM output handled (M5B3)
│   ├── test_edge_cases.py    # empty, long, emoji, non-English, sarcasm
│   └── golden/
│       ├── golden_set.json   # labelled feedback for accuracy (M5S1)
│       └── test_accuracy.py
│
├── sample_data/
│   └── feedback_sample.csv   # realistic demo data
│
└── docs/
    ├── DESIGN.md             # this document
    └── screenshots/
```

**Why this scales:** `core/` and `schemas/` have zero knowledge of Streamlit, FastAPI, or SQLite internals. You could add a Slack bot or a REST client tomorrow by writing one thin file. Prompts are versioned and separate from code, so changing a prompt is a data change, not a code change. Tests mirror the pipeline stages.

**Rubric impact:** M5E1 (readable, single-responsibility), M5B (each stage testable), M5B5 (`.env.example` + `.gitignore`), M5D5 (design maturity).

---

## 13. Git strategy

The rubric checks your commit history for *spread over two weeks* (M5D3) and evidence of *solving problems as you go* (M5D5). A single dump the night before is a visible fail.

**Branching:** simple `main` + short-lived feature branches (`feat/analysis-stage`, `fix/sarcasm-fewshot`), merged via small PRs (even solo — PRs are a great learning artifact and show process). Don't over-engineer with Gitflow.

**Commit cadence:** aim for **2–4 small, meaningful commits per working day**, each doing one thing. Commit when a stage passes its test, when you add a few-shot example, when a chart renders.

**Commit naming (Conventional Commits):**
```
feat: add combined per-item analysis call
fix: handle emoji-only feedback without LLM call
test: add consistency test for identical inputs
docs: document taxonomy edge cases
refactor: extract llm_client provider abstraction
chore: add .env.example and gitignore secrets
```

**Milestones (tags):** `v0.1-ingest`, `v0.2-analysis`, `v0.3-themes`, `v0.4-summary`, `v0.5-dashboard`, `v1.0-demo`. These map to the roadmap phases and make your progress legible at a glance.

**The learning-demonstrated trick:** commit your *mistakes and fixes* honestly. A commit like `fix: few-shot example for negation ("not bad" was scored negative)` is *worth points* — it's direct evidence of M5D5 ("encountered a problem, solved it independently"). Don't rewrite history to hide the learning.

**Rubric impact:** M5D3, M5D5, M5D1, M5E (process discipline).

---

## 14. README plan

The README is graded directly (M5E2 — the reviewer tries to run your project from it, cold). Every section has a job.

| Section | Why it matters | Rubric |
|---|---|---|
| **Project overview** (1 paragraph + the hero feature) | Reviewer knows in 20s what this is and why it's different | M5C1, M5A5 |
| **Who it's for / problem solved** | CX/PM/CS teams + the specific pain | M5A5, M5C1 |
| **Architecture diagram** (the §4 picture) | Shows the decoupled design at a glance | M5A3, M5E1 |
| **Setup** (prereqs, `.env` from `.env.example`, install) | The reviewer *must* be able to run it | M5E2, M5B5 |
| **Running** (Docker one-liner + Streamlit command) | Frictionless demo | M5E2, M5C4 |
| **Screenshots** (dashboard + grounded brief) | Proves it works even if their run hiccups | M5C4, M5S6 |
| **AI pipeline** (the §6 stages, plain English) | Demonstrates AI understanding in writing | M5A1, M5A3 |
| **Prompt strategy** (temp 0, few-shot rationale, versioning) | Shows deliberate design | M5S3, M5A2 |
| **Design decisions** (link to this DESIGN.md; the "no vector DB" call) | Senior-level judgment on display | M5A2, M5D5 |
| **Known limitations** (LLM confidence calibration, sarcasm residual) | Honesty = maturity | M5D4, M5A4 |
| **Future improvements** (Postgres, React UI, Emerging Radar) | Shows you know the ceiling | M5D2, M5D4 |
| **Lessons learned** (what broke, what surprised you) | Direct M5D fuel | M5D1, M5D5 |

**Golden rule:** have a friend clone and run it from the README with zero help *before* the demo. If they get stuck, the README is wrong. That single test is the best M5E2 insurance.

---

## 15. Dashboard plan

Designed so a VP of CX understands it in 30 seconds with no explanation (M5C2, M5S6). Every chart's title states the *question it answers*.

| Visualization | What it shows | Why an executive cares | Chart type |
|---|---|---|---|
| **Feedback volume (over time)** | Count per day/week | "Are complaints rising?" | Line/area (Plotly) |
| **Category distribution** | % per category | "Where are the problems concentrated?" | Horizontal bar (sorted) |
| **Sentiment trend** | Sentiment over time | "Is customer happiness moving?" | Stacked area / line |
| **Urgency distribution** | Count per urgency level | "How much is on fire right now?" | Bar, colour-coded (critical=red) |
| **Top recurring themes** | Ranked by Priority Score | "What are the *specific* top issues?" | Bar with theme labels |
| **Theme evolution** (nice-to-have) | Theme frequency week-over-week | "What's getting worse?" | Multi-line |
| **The Grounded Action Brief** ⭐ | Cited insights + ranked actions + expected impact | "What do I do Monday morning?" | Text panel with clickable quote links |
| **Recommended actions** | Ranked, with expected impact | "What's the ROI of fixing each?" | Ordered list / table |
| **Top customer quotes** | Representative verbatims per theme | "Show me real voices, not just numbers" | Quote cards (with IDs) |
| **Confidence / review queue** | Items the AI flagged for human review | "How much can I trust this, and what needs a human?" | Metric + filterable table |

**Layout principle:** top row = the *answer* (headline metrics + the Grounded Brief). Middle = the *evidence* (charts). Bottom = the *drill-down* (per-item table + review queue). Executive reads top-down and stops whenever satisfied.

**Interpretability rules (M5S6):** every axis labelled with units; every chart titled as a question; critical urgency always red; percentages *and* counts shown; no unexplained jargon in labels ("Checkout timeout on mobile," never "cluster_7").

**Rubric impact:** M5S6, M5C2, M5C3, M5S5, M5A4 (review queue visible), M5S4.

---

## 16. Testing plan

Testing is where you *prove* the reliability the rubric checks. Map each test to the rubric item it defends.

| Test class | What it does | Defends |
|---|---|---|
| **Golden dataset accuracy** | ~40–60 hand-labelled items; measure category/sentiment/urgency accuracy | M5S1 |
| **Consistency test** | Run identical input twice; assert equal outputs | M5B1, M5S2 |
| **Schema validation** | Feed malformed / partial LLM responses; assert graceful handling, no crash | M5B3, M5B4 |
| **Empty / whitespace input** | Assert it's skipped with a reason, not sent to LLM | M5B2 |
| **Very long input** | Assert truncation + flag, no crash | M5B2 |
| **Non-English input** | Assert language detected and handled/flagged | M5B2 |
| **Emoji-only input** | Assert graceful, meaningful response | M5B2 |
| **Sarcasm / negation** | Golden items like "oh great, another crash" → negative | M5S1, failure modes |
| **Mixed sentiment** | "love UI, hate billing" → `mixed` | M5S1 |
| **Spam / gibberish** | Routed to Other + low confidence + review queue | M5B2, M5A4 |
| **Duplicate feedback** | Content hash → served from cache, not re-analysed | M5B1, cost |
| **API failure simulation** | Mock the LLM to raise / return junk / invalid key → retry then safe fallback, no crash | M5B3 |
| **Cited-ID validation** | Summary insights citing non-existent IDs are dropped | Hero feature integrity |

**Expected behaviour is the point:** for each, write down *what should happen* before you test. "Empty input → skipped with logged reason, not an exception" is the assertion. This turns testing from a chore into rubric evidence.

**Regression tests:** when you bump a prompt version, re-run the golden set and compare accuracy to the previous version. This protects you from "I fixed sarcasm but broke billing" and is great M5D3 evidence.

**Rubric impact:** M5B1, M5B2, M5B3, M5B4, M5S1, M5S2, M5A4.

---

## 17. Accuracy evaluation

How to *measure* quality rather than assert it.

- **Classification accuracy:** on the golden set, overall accuracy + a **per-category confusion matrix**. The matrix shows *which* categories get confused (e.g., Performance↔Bug), which tells you which few-shot boundary to reinforce. Report precision/recall per category if you want depth — precision = "when it says Billing, how often right"; recall = "of all real Billing, how many it caught."
- **Sentiment consistency:** run each golden item 3× and measure agreement rate (should be ~100% at temp 0; anything less flags a determinism leak).
- **Urgency quality:** accuracy against labelled urgency, plus a check that `critical` is never *missed* on security/data-loss items (a false-negative on critical is the costliest error — weight it).
- **Theme quality (partly human eval):** are themes *specific* (M5S4)? Score each theme 1–5 for specificity; target avg ≥ 4. "DB timeout" = 5; "customer issues" = 1.
- **Summary usefulness (human eval):** the "read it aloud" test — could a VP act on it immediately? Rate coherence + actionability + grounding (do the cited quotes actually support the claim?).
- **Confidence calibration (the honest, senior move):** bucket predictions by reported confidence and check accuracy per bucket. If "0.9 confidence" items are only 70% accurate, your confidence is *overstated* — say so openly (M5D4) and use the empirical curve to set the review-queue threshold. This is exactly the kind of "research beyond the brief" that wins M5D5.
- **Blind mentor testing (M5S1):** you can't see these, so maximise your golden-set coverage of boundary cases and keep temp 0 so behaviour is predictable.

**Rubric impact:** M5S1, M5S2, M5S4, M5S5, M5A4, M5D4, M5D5.

---

## 18. Failure modes & mitigations

Name every realistic way the AI fails and how you defend against it. This section *is* your answer to M5A4.

| Failure mode | Why it happens | Mitigation |
|---|---|---|
| **Hallucinated category** | Model invents a label outside the taxonomy | Enum-constrained structured output; reject non-enum at parse; retry→Other+review |
| **Wrong sentiment (sarcasm)** | "oh great, love the crashes" reads positive lexically | LLM (not lexicon) + explicit sarcasm few-shot example |
| **Negation errors** | "not bad" → negative | Negation few-shot example; LLM context handling |
| **Mixed-sentiment collapse** | Two-sided feedback forced to one label | `mixed` label in the enum |
| **Theme explosion** | Too many tiny clusters | Min cluster size; merge small clusters; cap count |
| **Overgeneralised themes** | "customer issues" | Embeddings clustering + LLM instructed to be specific + specificity eval |
| **Hallucinated trend in summary** | LLM invents a pattern | Feed only deterministic aggregate; require ID citations; validate IDs exist |
| **Language ambiguity / non-English** | Model guesses | Language detection → flag/handle; edge-case test |
| **Typos / gibberish** | Low signal | Low confidence → review queue; Other fallback |
| **Code snippets / URLs in feedback** | Confuses the model | Conservative cleaning; robust to noise; still classify the human intent |
| **Emoji-only** | No text signal | Handle gracefully (emoji sentiment or route to review), never crash |
| **Duplicate / one loud customer** | Inflates a theme | Dedup by hash; `customer_id` reveals single-source spikes |
| **Overconfident model** | Poor calibration | Calibration analysis (§17); threshold from data, not vibes |
| **API failure / bad key / rate limit** | Network/vendor | Retry once → safe fallback → logged; never crash (M5B3) |
| **Malformed JSON response** | Rare model slip | Pydantic parse → retry → fallback |
| **Silent model update** | Vendor changes model | Pin model version; stamp on every output |
| **Non-deterministic scores** | temperature > 0 | temp 0 + cache by content hash |

**The one-liner for the demo:** *"My system's most likely wrong answers are sarcasm and typo-heavy feedback. I mitigate with targeted few-shot examples and a confidence gate that routes uncertain items to a human review queue — so a wrong answer becomes a flagged answer, not a silent one."* That answer alone can max M5A4.

---

## 19. Demo preparation

For each rubric item: the likely question and an ideal answer. Understand the *why* — don't memorise.

**M5A1 — "Explain your AI approach to me like I'm a PM."**
> "The system reads each piece of feedback and labels four things — what it's about, how the customer feels, how urgent it is, and the specific issue. It uses a large language model with carefully chosen examples so it labels consistently. Then it groups similar feedback into themes and writes a weekly brief that ranks issues by business impact — and every claim in that brief links to the real customer quotes behind it." *(Plain English, business-first, names the hero feature.)*

**M5A2 — "Why embeddings and not keyword search? Why no vector DB?"**
> "Keyword search can't tell that 'app keeps crashing' and 'constant force-closes' are the same issue — they share no words. Embeddings capture meaning, so those cluster together into one specific theme. I deliberately did *not* add a vector database: those exist to scale similarity search to millions of vectors across machines. My corpus is a few thousand items that fit in memory, so a vector DB would be complexity solving a problem I don't have." *(This is the flagship answer — the *decline* is the signal of maturity.)*

**M5A3 — "Walk me through input to output."**
> Point at the §6 diagram: ingest → validate → clean → one structured LLM call for the four labels → confidence gate → embed & cluster into themes → deterministic aggregation in code → grounded summary → dashboard.

**M5A4 — "Where is it most likely to be wrong?"**
> Use the one-liner from §18. Sarcasm + typos; mitigated by few-shot + confidence-gated review queue.

**M5A5 — "Who uses this and what problem does it solve?"**
> "CX and Product teams drowning in feedback across channels. It turns thousands of raw comments into a ranked, evidence-backed action list — so instead of reading everything, a PM opens Monday's brief and knows the top three things to fix and how many customers each affects."

**M5B1 / M5S2 — "Run the same input twice — do you get the same output?"**
> "Yes — temperature 0 makes the model near-deterministic, the model version is pinned, and identical text is served from cache by content hash. Same input, same output, by design."

**M5B2 — "What happens with empty / very long / non-English input?"**
> Demo each: empty → skipped with a reason; long → truncated + flagged; non-English → detected and flagged. "None of them crash; each has defined behaviour."

**M5B3 — "What if the API key is wrong?"**
> Show it: the call retries once, then returns a safe fallback and a clear message, and the dashboard still renders. "It degrades gracefully instead of crashing."

**M5B5 — "Show me your config / secrets."**
> Show `.env.example` (documents keys, no values), `.gitignore` (excludes `.env`), and that the code reads from environment. "No key ever touches the repo."

**M5S3 — "Why these few-shot examples?"**
> "Each targets a boundary the model got wrong in zero-shot testing — sarcasm, negation, Performance-vs-Bug, Pricing-vs-Billing. They're chosen from my own error analysis, not picked at random." *(Grounded in your process = top marks.)*

**M5S4 — "Are your themes specific?"**
> Show themes like "checkout timeout on mobile" and "duplicate billing charges," not "customer issues." "Specificity comes from clustering by meaning then having the model name each cluster from its actual members."

**M5S5 — "Read me the weekly summary — could a VP act on it?"**
> Read it. Each insight cites its supporting quotes; each action has an expected impact. "It's ranked, evidence-backed, and tells them exactly what to do first."

**M5D1 — "What was hardest?"** / **M5D2 — "What would you do differently?"** / **M5D4 — "What are you least confident about?"**
> Be honest and specific. Hardest: getting consistent themes without explosion / calibrating confidence. Differently: maybe start the golden set on day one. Least confident: LLM confidence calibration — "I measured it and know it's imperfect, which is why I have the review queue." *(Honesty is the rubric here — don't fake certainty.)*

**M5D5 — "What did you figure out that wasn't in the brief?"**
> "That LLM self-reported confidence is poorly calibrated, so I ran a calibration analysis and set my review threshold from data. And that a vector DB was a trap — I reasoned my way out of adding it."

**Rubric impact:** all of M5A, plus the reflection items of M5D.

---

## 20. Scope control

Explicit line between what you *must* ship and what's a bonus. Protects you from scope creep (M5C4 — "doesn't promise more than it delivers").

**REQUIRED (must work end-to-end):**
- Ingest → validate → clean → per-item analysis (category, sentiment, urgency)
- Confidence gate + review queue
- Embeddings theme clustering + specific labels
- Deterministic aggregation
- Grounded weekly summary with cited IDs (the hero)
- SQLite persistence with idempotent cache
- Streamlit dashboard with labelled charts + the brief
- Temp 0, pinned model, `.env`, retry/fallback
- Golden dataset + consistency + edge-case + API-failure tests
- README that runs cold + Dockerfile

**NICE-TO-HAVE (only if required is rock-solid):**
- FastAPI endpoints
- Emerging Issue Radar (needs multi-week data)
- Theme evolution chart
- Live Streamlit Cloud URL
- Expected-impact ("deflection") estimates on actions
- Customer_id "loud customer" detection

**CUT LINE:** if week 2 gets tight, ship every *required* item polished before touching any nice-to-have. A complete, reliable core beats a half-finished ambitious feature — that's literally what M5C4 measures.

---

## 21. Implementation roadmap

Twelve phases, each depending only on completed earlier ones. Build foundation → completion.

### Phase 1 — Project initialization
- **Goal:** runnable skeleton, secrets safe.
- **Files:** repo, `.gitignore`, `.env.example`, `requirements.txt`, `README` stub, `llm_client.py` (just a "hello LLM" call).
- **Learn:** env-based config; never committing secrets.
- **Acceptance:** one script makes a single successful LLM call reading the key from `.env`.
- **Common mistakes:** hardcoding the key; forgetting `.gitignore` for `.env`.
- **Checklist:** [ ] key in `.env` [ ] `.env` gitignored [ ] LLM call works.
- **Output:** a printed model response.
- **Commit:** `chore: project skeleton and env-based llm client`
- **Reflection:** where would a secret leak, and how did you prevent it?

### Phase 2 — Schemas & folder structure
- **Goal:** the data contracts (§8) before any logic.
- **Files:** everything in `schemas/`; the full folder tree.
- **Learn:** why typed schemas make everything downstream safer.
- **Acceptance:** Pydantic models validate a hand-written example object.
- **Common mistakes:** designing schemas *after* code (leads to churn).
- **Output:** importable, validating models.
- **Commit:** `feat: pydantic schemas for feedback, analysis, themes, report`
- **Reflection:** which field will you be tempted to drop, and why keep it?

### Phase 3 — Ingestion, validation, cleaning
- **Goal:** stages 1–3, no LLM yet.
- **Files:** `ingest.py`, `validate.py`, `clean.py`; `sample_data/feedback_sample.csv`.
- **Learn:** garbage-in prevention; edge-case handling as a *feature*.
- **Acceptance:** empty/long/duplicate/non-English handled per §16; tests pass.
- **Common mistakes:** over-aggressive cleaning that destroys meaning.
- **Commit:** `feat: ingestion + validation + cleaning with edge-case handling`
- **Reflection:** which edge case surprised you?

### Phase 4 — Prompt engineering (analysis)
- **Goal:** the system prompt + few-shot for the combined call.
- **Files:** `prompts/system_analysis_v1.txt`, `prompts/fewshot_analysis_v1.json`.
- **Learn:** zero-shot → error analysis → deliberate few-shot; temp 0.
- **Acceptance:** run zero-shot on ~20 items, record errors, add targeted examples.
- **Common mistakes:** stuffing many random examples instead of boundary ones.
- **Commit:** `feat: analysis prompt v1 with boundary few-shot examples`
- **Reflection:** which zero-shot error did each example fix?

### Phase 5 — Per-item analysis (LLM)
- **Goal:** stage 4 + confidence gate (stage 5).
- **Files:** `analyze.py`; retry/fallback logic.
- **Learn:** structured output, validation, graceful fallback.
- **Acceptance:** produces valid `PerItemAnalysis`; malformed responses don't crash; consistency test passes (same input twice → equal).
- **Common mistakes:** no retry/fallback → crashes on a bad response.
- **Commit:** `feat: combined per-item analysis with retry and confidence gate`
- **Reflection:** how does your fallback keep the pipeline alive?

### Phase 6 — Storage & caching
- **Goal:** SQLite + content-hash idempotency.
- **Files:** `storage/db.py`, `storage/cache.py`, `utils/hashing.py`.
- **Learn:** idempotency; why caching also guarantees consistency & saves cost.
- **Acceptance:** re-running never re-analyses a seen item; results persist.
- **Commit:** `feat: sqlite persistence with content-hash cache`
- **Reflection:** how does the cache key relate to prompt versioning?

### Phase 7 — Theme extraction
- **Goal:** stage 6 — embeddings → cluster → LLM labels.
- **Files:** `ai/embeddings.py`, `core/themes.py`.
- **Learn:** semantic similarity; why not a vector DB; theme-explosion control.
- **Acceptance:** themes are *specific* (specificity eval ≥ 4 avg); no explosion.
- **Common mistakes:** reaching for a vector DB; unbounded cluster counts.
- **Commit:** `feat: embedding-based theme clustering with llm labelling`
- **Reflection:** show one theme keywords would have missed.

### Phase 8 — Aggregation
- **Goal:** stage 7 — deterministic counts + priority scores.
- **Files:** `core/aggregate.py`.
- **Learn:** never let the LLM count; code does maths.
- **Acceptance:** distributions exact and identical on re-run.
- **Commit:** `feat: deterministic aggregation and priority scoring`
- **Reflection:** why must this stage be code, not LLM?

### Phase 9 — Weekly summary (hero)
- **Goal:** stage 8 — grounded, cited exec brief.
- **Files:** `prompts/system_summary_v1.txt`, `core/summarize.py` (+ ID validation).
- **Learn:** map-reduce summarisation; grounding as anti-hallucination.
- **Acceptance:** every insight cites real IDs; fake-ID insights dropped; summary is coherent + actionable.
- **Commit:** `feat: grounded action brief with cited feedback ids`
- **Reflection:** how do you *prove* it isn't hallucinating?

### Phase 10 — Dashboard
- **Goal:** stage 9 — Streamlit + Plotly + the brief.
- **Files:** `dashboard/app.py`.
- **Learn:** interpretable charts; non-technical usability.
- **Acceptance:** a non-technical person navigates it unassisted; charts self-explain; quotes clickable.
- **Common mistakes:** unlabelled axes; cluster IDs shown instead of theme names.
- **Commit:** `feat: streamlit dashboard with labelled charts and grounded brief`
- **Reflection:** what did a test user find confusing?

### Phase 11 — Testing & evaluation
- **Goal:** golden set, consistency, edge, API-failure, calibration.
- **Files:** everything in `tests/`, `golden_set.json`.
- **Learn:** measuring quality; confidence calibration.
- **Acceptance:** all §16 tests pass; accuracy + confusion matrix + calibration recorded.
- **Commit:** `test: golden accuracy, consistency, edge cases, api failure`
- **Reflection:** which category confuses most, and your fix?

### Phase 12 — Documentation & demo prep
- **Goal:** cold-runnable README, Dockerfile, screenshots, rehearsed answers.
- **Files:** finished `README.md`, `Dockerfile`, `docs/screenshots/`.
- **Learn:** documentation as a deliverable; articulating decisions.
- **Acceptance:** a friend runs it from the README with zero help; you can answer every §19 question.
- **Commit:** `docs: complete readme, dockerfile, and demo screenshots`
- **Reflection:** what would you do differently if you started today?

---

## 22. Glossary

- **Few-shot prompting** — giving the model a handful of example input→output pairs to pin down desired behaviour. *Zero-shot* = no examples.
- **Temperature** — randomness of the model's output. 0 = deterministic (always the most likely token). Essential for consistency.
- **Structured outputs / tool use** — forcing the model to return JSON matching a strict schema, so parsing never guesses.
- **Embeddings** — numeric vectors capturing meaning; similar text → nearby vectors. Enables semantic clustering.
- **Clustering** — grouping vectors by similarity (here, to form themes). *HDBSCAN* auto-picks cluster count and marks noise.
- **RAG (retrieval-augmented generation)** — retrieving relevant documents to ground an LLM's answer. *Not used here* — there's nothing to retrieve for per-item labelling.
- **Grounding / attribution** — tying every generated claim to real source data (feedback IDs), preventing hallucination.
- **Idempotency** — running the same operation twice has the same effect as once (here, via content-hash caching).
- **Calibration** — whether a model's stated confidence matches its real accuracy. LLM confidence is often *over*stated.
- **Map-reduce summarisation** — aggregate deterministically (reduce) then summarise the aggregate, instead of dumping raw data into one prompt.
- **Priority Score** — transparent ranking = volume × urgency × negativity, normalised. Turns themes into an action order.

---

*End of design document. Implementation should now be an engineering exercise, not a design one.*

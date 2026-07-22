# PulseAI — Customer Feedback Intelligence Platform

> Status: **Phase 1 — project initialization.** This README is a stub;
> the full version (architecture, screenshots, lessons learned) is built
> out in Phase 12 per `docs/DESIGN.md`.

## What this is

An AI-powered platform that classifies customer feedback, scores sentiment
and urgency, discovers recurring themes, and generates a weekly executive
brief where every insight cites the real feedback behind it. Full design
rationale: see [`docs/DESIGN.md`](docs/DESIGN.md).

## Setup (Phase 1)

```bash
git clone <this-repo>
cd pulseai
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your real OPENAI_API_KEY
```

## Verify your connection

```bash
PYTHONPATH=src python -m pulseai.ai.llm_client
```

Expected output:
```
Using model: gpt-5-nano (reasoning model: True)
Model replied: PulseAI connection OK
```

If it fails, you'll see a clear error (bad key, network issue) rather than
a crash — the client's retry-then-error-message path is intentional
(see `docs/DESIGN.md` §9, §18).

## Project layout

See `docs/DESIGN.md` §12 for the full annotated folder structure and the
`docs/DESIGN.md` §21 roadmap for what gets built in each phase.

## Roadmap status

- [x] Phase 1 — Project initialization (this commit)
- [ ] Phase 2 — Schemas & folder structure
- [ ] Phase 3 — Ingestion, validation, cleaning
- [ ] Phase 4 — Prompt engineering (analysis)
- [ ] Phase 5 — Per-item analysis (LLM)
- [ ] Phase 6 — Storage & caching
- [ ] Phase 7 — Theme extraction
- [ ] Phase 8 — Aggregation
- [ ] Phase 9 — Weekly summary (hero feature)
- [ ] Phase 10 — Dashboard
- [ ] Phase 11 — Testing & evaluation
- [ ] Phase 12 — Documentation & demo prep

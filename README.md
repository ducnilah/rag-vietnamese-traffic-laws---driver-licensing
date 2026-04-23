# Traffic Law & Driver Licensing Assistant (RAG)

A milestone-driven RAG chatbot project for Vietnamese traffic law and driver licensing assistance.

## Architecture

The project follows three layers:

1. Offline pipeline: ingest, clean, chunk, index, quality checks.
2. Online pipeline: query understanding, hybrid retrieval, reranking, context building, answer generation.
3. Ops/safety/evaluation: caching, guardrails, logs/tracing, retrieval and faithfulness metrics.

## Milestones

1. `M1` Offline indexing foundation: parse -> chunk -> BM25 -> quality checks.  
Status: implemented.
2. `M2` Online retrieval foundation: hybrid retrieval API, BM25 + dense (ChromaDB target), fusion/rerank, retrieval tests.  
Status: implemented baseline.
3. `M3` Context builder: citation map, confidence scoring, neighbor/overlap chunk expansion, token-bounded context assembly.
4. `M4` Conversation persistence: thread/message storage, short-term + long-term memory, summary pipeline.
5. `M5` LLM integration and guardrails: grounded generation, policy checks, OOD/hallucination controls.
6. `M6` Evaluation harness and regression: retrieval metrics, faithfulness checks, latency tracking, regression suite.
7. `M7` Production readiness: auth/session integration hooks, observability, migration/runbook hardening.

## Locked Tech Stack

- Backend API: `FastAPI`
- Relational DB (app state): `PostgreSQL`
- ORM / data access: `SQLAlchemy`
- DB migrations: `Alembic`
- Vector DB (dense retrieval): `ChromaDB`
- Lexical retrieval: `BM25`
- Retrieval strategy: `Hybrid BM25 + Dense + Fusion/Rerank`
- Frontend: `Next.js + TailwindCSS`

## Data Folder Convention

- `data/raw/pdf`: original PDF sources.
- `data/raw/docx`: original DOCX sources.
- `data/raw/txt_md`: manually prepared text/markdown sources.
- `data/processed/text`: normalized artifacts (including table-aware markdown outputs).
- `data/index`: offline indexing artifacts (documents/chunks/BM25/quality/chroma persist dir when enabled).

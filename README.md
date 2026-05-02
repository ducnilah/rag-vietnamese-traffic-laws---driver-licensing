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
Status: implemented baseline.
4. `M4` Conversation persistence: thread/message storage, short-term + long-term memory, summary pipeline.
Status: implemented baseline.
5. `M5` LLM integration and guardrails: grounded generation, policy checks, OOD/hallucination controls.
Status: implemented baseline.
6. `M6` Evaluation harness and regression: retrieval metrics, faithfulness checks, latency tracking, regression suite.
7. `M7` Production readiness:
Status: in progress.
7.1 `Auth + Ownership` (implemented):
- register/login/me endpoints
- bearer token verification
- strict thread ownership checks on all thread/message/chat endpoints
- DB migration + seed flow for users
7.2 `Chat Session API UX` (implemented):
- rename/archive thread (`PATCH /threads/{thread_id}`)
- list threads with/without archived (`include_archived`)
- message pagination with cursor (`before_message_id`)
- API responses shaped for chat UI consumption (`messages` + `page`)
7.3 `Web App UX` (planned next):
- Next.js + Tailwind app
- login/register screens + protected routes
- chat layout (sidebar threads + message pane)
- resume old threads and continue chat naturally

## Locked Tech Stack

- Backend API: `FastAPI`
- Relational DB (app state): `PostgreSQL`
- ORM / data access: `SQLAlchemy`
- DB migrations: `Alembic`
- Vector DB (dense retrieval): `ChromaDB`
- Lexical retrieval: `BM25`
- Retrieval strategy: `Hybrid BM25 + Dense + Fusion/Rerank`
- LLM runtime (M5 default): `Qwen2.5 1.5B Instruct` via `Ollama`
- Frontend: `Next.js + TailwindCSS`

## Data Folder Convention

- `data/raw/pdf`: original PDF sources.
- `data/raw/docx`: original DOCX sources.
- `data/raw/txt_md`: manually prepared text/markdown sources.
- `data/processed/text`: normalized artifacts (including table-aware markdown outputs).
- `data/index`: offline indexing artifacts (documents/chunks/BM25/quality/chroma persist dir when enabled).

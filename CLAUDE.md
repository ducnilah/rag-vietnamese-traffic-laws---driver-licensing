# CLAUDE.md

This file is the working plan for `traffic-law-v2`, a new implementation of the Vietnamese traffic law and driver licensing RAG assistant.

## Project Intent

V2 keeps the same product domain as v1:
- Vietnamese traffic law Q&A.
- Driver licensing, training, testing, and GPLX procedures.
- Citation-backed legal answers.
- Multi-turn chatbot UX with login, saved threads, and conversation continuity.

The implementation direction changes:
- v1 was mostly custom-built RAG orchestration.
- v2 uses LangChain for orchestration, retrievers, prompt templates, structured output, and trace hooks.
- v2 uses external/API models for LLM and embeddings instead of local Ollama models as the default path.
- v2 keeps the strongest v1 lessons: table-aware parsing, hybrid retrieval, citation maps, conversation memory, permissive normal chat, and strict legal grounding.

## Lessons From V1

Keep:
- Table-aware parsing for PDF/DOCX legal documents.
- Legal metadata: document title, chapter, section, article, page, table id, source version.
- Hybrid retrieval: lexical BM25 plus dense vector retrieval.
- Context builder with token budget, citation map, confidence score, deduplication, and optional neighbor expansion.
- Postgres-backed app state for users, threads, messages, summaries, and user memory.
- Frontend/API separation.
- Tests at every milestone.

Improve:
- Replace weak local generation with stronger hosted/API LLMs.
- Use API embeddings by default instead of CPU-heavy local embedding builds.
- Make retrieval and generation easier to trace and evaluate.
- Keep API handlers thin; move orchestration into services/chains.
- Improve natural conversation: model should respond like an assistant, not only a document reader.
- Use explicit evidence policy: personal/conversational questions can use chat memory; legal claims require retrieved evidence.
- Add evaluation earlier so retrieval/prompt changes can be compared, not guessed.

## Target Architecture

Offline / indexing:
- Load source documents from `data/raw`.
- Parse PDF/DOCX/MD/TXT into normalized text and table-aware markdown.
- Extract legal metadata: `source_id`, `document_title`, `legal_type`, `issuing_body`, `effective_date`, `chapter`, `section`, `article`, `page`, `table_id`, `version`.
- Chunk by legal structure first, token budget second.
- Build vector index, BM25 index, docstore, and indexing quality report.

Online / serving:
- Receive user query in an authenticated thread.
- Build conversational input from recent messages, thread summary, and user memory.
- Classify query mode: conversational, legal/retrieval-needed, mixed, or policy-risk.
- For retrieval-needed queries: rewrite/focus query, run hybrid retrieval, rerank/compress, build citation map.
- Generate answer with hosted/API LLM through LangChain.
- Persist user message, assistant message, citations, confidence, model metadata, and trace id.

## Runtime Stack

- API: FastAPI.
- RAG orchestration: LangChain LCEL / Runnable chains.
- LLM provider: OpenAI-compatible by default, provider-agnostic where practical.
- Embeddings: API embedding model by default.
- Vector DB: Chroma for local/dev, Qdrant as production-ready option.
- Lexical retrieval: BM25.
- State DB: Postgres + SQLAlchemy + Alembic.
- Frontend: Next.js + Tailwind, conceptually carried forward from v1.
- Observability: LangSmith optional; structured local traces required.

## Locked Defaults For V2

- API model path is default. No local Ollama dependency in the main flow.
- `MODEL_PROVIDER=openai_compatible` initially.
- `OPENAI_BASE_URL=https://openrouter.ai/api/v1` initially.
- `LLM_MODEL=google/gemini-3.1-flash-lite` initially, adjustable by env.
- `EMBEDDING_PROVIDER=local` initially.
- `EMBEDDING_MODEL=BAAI/bge-m3` initially, adjustable by env.
- Use LangChain for prompt templates, chains, retriever composition, and output parsers.
- Keep domain-specific legal parsing, metadata, citation formatting, table parsing, and hybrid scoring explicit.

## Milestones

### M1 - Project Bootstrap
Goal: create a clean v2 codebase that can run independently.

Scope:
- `pyproject.toml` with LangChain/API-model stack.
- `.env.example` for provider/model/database settings.
- `src/traffic_law_v2` package.
- FastAPI app factory.
- `GET /api/v1/health`.
- Config loader using `pydantic-settings`.
- Baseline health test.

Acceptance:
- App imports cleanly.
- Health endpoint returns provider/model config.
- Tests pass locally.

### M2 - Document Ingestion And Legal Metadata
Goal: rebuild ingestion with v1 lessons but cleaner boundaries.

Scope:
- Source folders: `data/raw/pdf`, `data/raw/docx`, `data/raw/txt_md`, `data/processed`, `data/index`.
- Parser interfaces for PDF/DOCX/MD/TXT.
- Table-aware conversion retained from v1 concept.
- Legal metadata schema as typed Pydantic models.
- Chunk model with stable ids and metadata.
- Quality report for empty chunks, oversized chunks, duplicate chunks, and bad tables.

LangChain usage:
- Use LangChain document abstractions where helpful.
- Do not depend on LangChain loaders if they make table extraction worse.

Acceptance:
- Sample documents parse into normalized document objects.
- Chunks preserve article/table/page metadata.
- Quality report is generated.

### M3 - Embeddings And Vector Store
Goal: index chunks with API embeddings and a vector store.

Scope:
- LangChain embedding wrapper based on configured provider.
- Chroma local vector store first.
- Optional Qdrant adapter planned for deployment.
- Docstore mapping chunk ids to canonical text and metadata.
- Index command with rebuild/incremental mode.

Acceptance:
- Index command writes vector store and docstore.
- Retrieval by semantic query returns chunk metadata and citations.
- API key absence fails clearly for indexing/generation paths.

### M4 - Hybrid Retrieval And Reranking
Goal: preserve v1 retrieval quality while using LangChain composition.

Scope:
- Dense retriever from vector store.
- BM25 retriever.
- Hybrid fusion with reciprocal rank fusion or weighted normalized fusion.
- Table intent boost.
- Diversity/dedup.
- Optional reranker/compressor using LangChain contextual compression or provider reranker later.

Acceptance:
- Legal/table queries retrieve table chunks when appropriate.
- Traffic penalty queries are not incorrectly blocked before retrieval.
- Retrieval diagnostics endpoint exposes candidates and scores.

### M5 - Context Builder And Citation Contract
Goal: produce deterministic context packages for the LLM.

Scope:
- Context builder receives retrieved chunks and conversation memory.
- Token budget handling.
- Neighbor expansion for legal continuity.
- Citation map with `C1`, `C2`, document title, article/chapter/page/table id, chunk id, score.
- Citation coverage score and confidence score.

Acceptance:
- Every legal answer can map claims back to citation ids.
- Context package is testable without calling the LLM.

### M6 - LangChain Generation Chain
Goal: use hosted/API LLMs effectively and naturally.

Scope:
- LangChain `ChatPromptTemplate` for system identity, answer style, evidence policy, conversation memory, retrieved context.
- Structured output parser where practical.
- Answer modes: conversational, grounded legal answer, insufficient evidence.
- Natural Vietnamese by default, no random foreign-language mixing, concise unless user asks for detail.

Acceptance:
- “hi” and personal follow-up questions feel natural.
- Legal answers include citations.
- If context is weak, model says so instead of fabricating.

### M7 - Conversation State And Memory
Goal: product-grade chat continuity.

Scope:
- Postgres models: users, threads, messages, thread_summaries, user_memory, retrieval_traces, answer_feedback.
- Memory extraction chain for name, age, target license class, and location if relevant.
- Thread summary chain after N messages.
- Auth and ownership enforcement.

Acceptance:
- User can leave and return to an old thread.
- Bot can answer “what is my name?” from memory.
- Legal questions still use retrieval, not only memory.

### M8 - Evaluation And Regression
Goal: measure quality before polishing.

Scope:
- Golden question set for licensing requirements, age requirements, training hours tables, traffic penalties, document/procedure questions.
- Metrics: retrieval hit rate, MRR/nDCG, citation correctness, faithfulness, helpfulness, latency, cost per answer.
- Regression command runnable locally.

Acceptance:
- Evaluation report generated as JSON/Markdown.
- Retrieval/prompt changes can be compared against baseline.

### M9 - Frontend And Product Polish
Goal: rebuild product UX once backend is stable.

Scope:
- Next.js chat UI.
- Login/register.
- Thread sidebar.
- Citation cards.
- Confidence/evidence display.
- Feedback buttons for answer quality.

Acceptance:
- End-to-end chat works from UI.
- Citations are readable.
- Old conversations resume correctly.

### M10 - Deployment Readiness
Goal: prepare for hosted demo/deployment.

Scope:
- Dockerfile and compose.
- Environment matrix.
- Database migration flow.
- Vector index build/runbook.
- Secret handling.
- Cost controls and rate limit behavior.
- Basic observability.

Acceptance:
- Fresh environment can be brought up from documented steps.
- Indexing and API serving are separate, repeatable commands.

## API Contract Direction

Base path: `/api/v1`

Core endpoints:
- `GET /health`
- `POST /indexes/build`
- `POST /retrieve`
- `POST /context`
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- `POST /threads`
- `GET /threads`
- `GET /threads/{thread_id}/messages`
- `POST /threads/{thread_id}/chat`
- `POST /feedback`

Response requirements:
- Legal answers return `answer`, `citations`, `confidence`, `model`, `trace_id`.
- Errors use a consistent envelope.
- Debug retrieval endpoints are allowed in dev but must be gated in production.

## Testing Strategy

- Config and API bootstrap tests in M1.
- Parser/chunker/metadata tests in M2.
- Vector retrieval tests in M3.
- Hybrid/rerank tests in M4.
- Context/citation tests in M5.
- Prompt chain tests with mocked LLM in M6.
- State/memory/API ownership tests in M7.
- Evaluation dataset in M8.

LLM/API calls should be mocked in unit tests. Real provider calls belong in manual or integration tests.

## Working Rules For V2

- Keep v2 independent from v1. Reference v1 for lessons, not imports.
- Prefer LangChain where it reduces orchestration complexity.
- Keep domain-specific legal parsing, metadata, and citation code explicit.
- Do not add local model dependencies to the default path.
- Every milestone should end with a runnable command and at least one test.
- Keep explanations in Vietnamese when discussing with the project owner.

## Build Status - 2026-05-22

Completed runnable backend slice from M1 to M10:
- M1 bootstrap: `pyproject.toml`, config, FastAPI health, tests.
- M2 ingestion: DOCX/TXT/MD loader, legal metadata, legal-structure chunking.
- M3 indexing: JSONL docstore, BM25, Chroma vector store, quality report.
- M4 retrieval: hybrid sparse+dense retrieval, dedup, table/penalty intent boost, vehicle-aware rerank.
- M5 context: citation map, context text, confidence.
- M6 generation: LangChain `ChatPromptTemplate`, OpenAI/OpenAI-compatible config, fallback generator for local no-key mode.
- M7 state: dev register/login/me, thread messages, simple memory extraction for name/age.
- M8 evaluation: `scripts/evaluate.py` with golden smoke cases.
- M9 frontend: minimal Next.js chat scaffold in `frontend/`.
- M10 deployment: `Dockerfile`, `docker-compose.yml`, `deployment/RUNBOOK.md`.

Current local index:
- Source: `data/raw/xu_phat_long.docx`.
- Index output: `data/index`.
- Last build: 1 document, 155 chunks, 0 quality warnings, 155 Chroma vectors.

Current test baseline:
- `python -m pytest -q`: 7 tests passing.
- `python scripts/evaluate.py --index-dir data/index`: 5/5 retrieval smoke cases passing.

Important caveat:
- Without a real API key, generation uses a deterministic fallback. This is enough for local pipeline stability, but final answer quality should be validated again with the chosen hosted LLM and embedding model.

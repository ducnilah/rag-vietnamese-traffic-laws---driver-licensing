# Traffic Law V2

LangChain-first RAG assistant for Vietnamese traffic law and driver licensing.

V2 keeps the domain and product lessons from v1, but changes the engine:
- LangChain for orchestration.
- API LLMs and API embeddings by default.
- Explicit citation contracts and evaluation from early milestones.

## Current Build

Implemented end-to-end backend slice:
- M1: project bootstrap, config, FastAPI health.
- M2: DOCX/TXT/MD ingestion, legal metadata, legal-structure chunking.
- M3: BM25 + Chroma vector index, deterministic fallback embeddings for local tests.
- M4: hybrid retrieval, vehicle-aware rerank, table/penalty intent boost, dedup.
- M5: context builder, citation map, confidence score.
- M6: LangChain prompt chain with OpenAI/OpenAI-compatible provider support; local fallback for no API key.
- M7: dev auth/register/login, in-memory thread state, messages, simple user memory.
- M8: retrieval regression command.
- M9: minimal Next.js chat UI scaffold in `frontend/`.
- M10: Dockerfile, compose, local runbook.

## Startup Setup

V2 currently stores dev users, threads, messages, memory, and feedback in `InMemoryState`.
PostgreSQL is configured in `.env`, but it is not required for the current web chat flow until persistent state is wired in.

### One-time setup

Run this once after cloning the project or recreating the environment:

```bash
cd "/Users/m1/Documents/rag-thesis1/traffic-law-v2"
cp .env.example .env
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Set the model provider in `.env`. Current OpenRouter-style setup:

```env
MODEL_PROVIDER=openai_compatible
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=your_openrouter_key
LLM_MODEL=google/gemini-3.1-flash-lite
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=4
```

Build the RAG index once, or whenever source documents change:

```bash
cd "/Users/m1/Documents/rag-thesis1/traffic-law-v2"
PYTHONPATH=src .venv/bin/python scripts/build_index.py --raw-dir data/raw --index-dir data/index
```

Install frontend dependencies once if `frontend/node_modules` is missing:

```bash
cd "/Users/m1/Documents/rag-thesis1/traffic-law-v2/frontend"
/usr/local/bin/node /usr/local/lib/node_modules/npm/bin/npm-cli.js install
```

### Daily startup

Start backend:

```bash
cd "/Users/m1/Documents/rag-thesis1/traffic-law-v2"
PYTHONPATH=src .venv/bin/python scripts/run_api.py
```

Start frontend in another terminal:

```bash
cd "/Users/m1/Documents/rag-thesis1/traffic-law-v2/frontend"
/usr/local/bin/node ./node_modules/next/dist/bin/next dev --port 3001
```

Open:

```text
http://localhost:3001
```

Use `/usr/local/bin/node` for the frontend on this Mac. Running Next with Codex's bundled Node can fail to load the native SWC compiler.

### Health check

```bash
curl http://127.0.0.1:8010/api/v1/health
```

Retrieval eval:

```bash
python scripts/evaluate.py --index-dir data/index
```

Chat smoke:

```bash
python scripts/chat_smoke.py \
  --query "Đi ngược chiều trên đường một chiều bị phạt thế nào?" \
  --index-dir data/index
```

## Model Config

Without `OPENAI_API_KEY`, local tests use hash embeddings and a fallback generator.
For the current OpenRouter model flow:

```env
MODEL_PROVIDER=openai_compatible
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=...
LLM_MODEL=google/gemini-3.1-flash-lite
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=4
```

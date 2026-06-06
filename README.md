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
- M7: auth/register/login, Postgres-backed thread state, messages, tokens, simple user memory.
- M8: retrieval regression command.
- M9: minimal Next.js chat UI scaffold in `frontend/`.
- M10: Dockerfile, compose, local runbook.

## Startup Setup

V2 stores users, login tokens, chat threads, messages, memory, and feedback in PostgreSQL when `DATABASE_URL` is set. `InMemoryState` is only for isolated tests via `DATABASE_URL=memory`.

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


### Database check

The default database URL is:

```env
DATABASE_URL=postgresql+psycopg://postgres:123456@localhost:5432/traffic_rag_v2
```

Backend startup creates these tables if they do not exist: `users`, `auth_tokens`, `chat_threads`, `chat_messages`, `user_memory`, `feedback`.

Check from terminal with `psql`:

```bash
export PGPASSWORD=123456
psql -h localhost -p 5432 -U postgres -d traffic_rag_v2
```

Useful checks inside `psql`:

```sql
\dt
SELECT id, email, created_at FROM users ORDER BY created_at DESC LIMIT 5;
SELECT id, user_id, title, created_at FROM chat_threads ORDER BY created_at DESC LIMIT 5;
SELECT thread_id, role, left(content, 80), created_at FROM chat_messages ORDER BY created_at DESC LIMIT 10;
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


## Evaluation Overview

The project now has an initial automatic eval setup for BERTScore and a deterministic RAG Triad overview.

Dataset:

```text
eval/datasets/traffic_law_eval.jsonl
```

Run a dry-run without calling the LLM. This uses reference answers as generated answers and is useful for checking retrieval/report plumbing:

```bash
cd "/Users/m1/Documents/rag-thesis1/traffic-law-v2"
EMBEDDING_PROVIDER=hash PYTHONPATH=src .venv/bin/python scripts/run_eval_overview.py \
  --skip-generation \
  --hf-offline \
  --dataset eval/datasets/traffic_law_eval.jsonl \
  --index-dir data/index \
  --out-dir eval/outputs
```

Run the real generation eval with the configured LLM and embedding provider:

```bash
cd "/Users/m1/Documents/rag-thesis1/traffic-law-v2"
PYTHONPATH=src .venv/bin/python scripts/run_eval_overview.py \
  --hf-offline \
  --dataset eval/datasets/traffic_law_eval.jsonl \
  --index-dir data/index \
  --out-dir eval/outputs
```

Outputs:

```text
eval/outputs/eval_overview_results.jsonl
eval/outputs/eval_overview_summary.json
eval/reports/eval_overview.md
eval/notebooks/01_eval_overview.ipynb
```

BERTScore needs the optional package `bert-score` and a transformer model. Use `--hf-offline` after the BERTScore model has been downloaded once, so Transformers reads from local cache instead of calling HuggingFace during eval. If it is not installed, the script still runs RAG Triad and reports BERTScore as unavailable. Install optional analysis dependencies when network is available:

```bash
source .venv/bin/activate
python -m pip install bert-score pandas matplotlib ipykernel
```

RAG Triad overview currently uses reproducible heuristic scoring:

```text
context_relevance = retrieved context matches the query
groundedness = answer terms are supported by retrieved context
answer_relevance = answer matches the reference answer
```

Manual evaluation is planned as a separate layer after this automatic overview.

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

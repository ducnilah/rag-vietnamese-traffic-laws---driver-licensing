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

## Run

```bash
cp .env.example .env
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/build_index.py --raw-dir data/raw --index-dir data/index
python scripts/run_api.py
```

Health:

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

Without `OPENAI_API_KEY`, local tests use hash embeddings and a fallback generator. With an API key, LangChain calls the configured provider:

```env
MODEL_PROVIDER=openai
OPENAI_API_KEY=...
LLM_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-large
```

OpenAI-compatible providers:

```env
MODEL_PROVIDER=openai_compatible
OPENAI_BASE_URL=https://provider.example/v1
OPENAI_API_KEY=...
```

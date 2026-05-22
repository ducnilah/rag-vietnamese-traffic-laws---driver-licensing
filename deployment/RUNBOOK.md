# Traffic Law V2 Runbook

## Local Backend

```bash
cd /Users/m1/Documents/rag-thesis1/traffic-law-v2
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

Chat smoke:

```bash
curl -X POST http://127.0.0.1:8010/api/v1/threads \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Model Provider

Without `OPENAI_API_KEY`, the project uses deterministic hash embeddings and a fallback answer template. This keeps tests and local retrieval stable.

For real answers:

```bash
MODEL_PROVIDER=openai
OPENAI_API_KEY=...
LLM_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-large
```

For OpenAI-compatible providers, set:

```bash
MODEL_PROVIDER=openai_compatible
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_API_KEY=...
```

## Rebuild Index

Rebuild whenever files in `data/raw` change:

```bash
python scripts/build_index.py --raw-dir data/raw --index-dir data/index
```

## Evaluation

```bash
python scripts/evaluate.py --index-dir data/index
```

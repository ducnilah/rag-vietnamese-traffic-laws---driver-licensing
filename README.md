# Traffic Law & Driver Licensing Assistant (RAG)

A step-by-step implementation of a production-style RAG chatbot for traffic law and driver licensing support.

## Architecture

The project follows three layers:

1. Offline pipeline: ingest, clean, chunk, index, quality checks.
2. Online pipeline: query understanding, hybrid retrieval, reranking, context building, answer generation.
3. Ops/safety/evaluation: caching, guardrails, logs/tracing, retrieval and faithfulness metrics.

## Milestones

1. `M1` Offline indexing foundation with tests. (Implemented)
2. `M2` Online retrieval API (hybrid BM25 + dense skeleton) with tests.
3. `M3` Context builder, citation map, and confidence scoring.
4. `M4` LLM integration and guardrails.
5. `M5` Evaluation harness and regression tests.

## Quick Start

```bash
python3 -m unittest discover -v -s tests -p "test_*.py"
python3 scripts/run_offline_index.py --raw-dir data/raw/txt_md --extra-dir data/processed/text --out-dir data/index --log-level INFO
python3 scripts/convert_pdf_table_aware.py --pdf data/raw/pdf/<your_file>.pdf
python3 scripts/search_index.py --query "cho tôi bảng số giờ học hạng A1" --index-dir data/index --log-level INFO
```

## Current Scope (M1)

- Parse text/markdown files from `data/raw/txt_md` + `data/processed/text`
- Clean and normalize content
- Create metadata-aware chunks
- Preserve table structure into dedicated `content_type=table` chunks
- Run quality checks on chunk outputs
- Build BM25 index artifact
- Export artifacts to `data/index`

## Data Folder Convention

- `data/raw/pdf`: put original `.pdf` files here.
- `data/raw/docx`: put original `.docx` files here.
- `data/raw/txt_md`: optional manual-converted `.txt/.md` files (used by current M1 parser).
- `data/processed/text`: parsed/cleaned text artifacts (next milestone).
- `data/index`: retrieval artifacts (`documents.jsonl`, `chunks.jsonl`, `bm25.json`, `quality_report.json`).

## Table-Aware PDF Output

When converting legal PDFs containing tables:

- `scripts/convert_pdf_table_aware.py` writes
  - `data/processed/text/<pdf_stem>_table_aware.md` (page text + markdown tables)
  - `data/processed/text/<pdf_stem>_tables.jsonl` (one structured table per line)
- This helps retrieval for questions asking exact values from rows/columns.

## Retrieval Priority

- The index now stores `content_type` in chunk metadata (`text` or `table`).
- `scripts/search_index.py` applies a table-priority rerank when query intent indicates table lookup.
- Search output now includes formatted legal citations (e.g., `Điều`, `Chương`, `Thông tư/Nghị định` when available).

## Logging

- `run_offline_index.py` and `search_index.py` support `--log-level` (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
- Use `--log-level DEBUG` to print per-step details while troubleshooting pipeline behavior.

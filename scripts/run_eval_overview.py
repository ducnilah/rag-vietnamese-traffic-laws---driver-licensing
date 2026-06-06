from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if "--hf-offline" in sys.argv:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
from typing import Any

from traffic_law_v2.context import build_context
from traffic_law_v2.eval.bertscore_metrics import compute_bertscore
from traffic_law_v2.eval.dataset import load_eval_dataset, write_jsonl
from traffic_law_v2.eval.rag_triad import compute_expected_checks, compute_rag_triad
from traffic_law_v2.eval.report import summarize, write_markdown_report
from traffic_law_v2.generation import generate_answer
from traffic_law_v2.retrieval import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval overview with BERTScore and RAG Triad.")
    parser.add_argument("--dataset", default="eval/datasets/traffic_law_eval.jsonl")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--out-dir", default="eval/outputs")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bertscore-model", default="bert-base-multilingual-cased")
    parser.add_argument("--skip-generation", action="store_true", help="Use reference answers as generated answers for pipeline dry-run.")
    parser.add_argument("--hf-offline", action="store_true", help="Use local HuggingFace cache only for BERTScore/model loading.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_dir = Path(args.out_dir)
    report_dir = Path("eval/reports")
    items = load_eval_dataset(dataset_path)
    retriever = HybridRetriever(Path(args.index_dir))

    rows: list[dict[str, Any]] = []
    candidates: list[str] = []
    references: list[str] = []
    for item in items:
        hits = retriever.retrieve(item.query, top_k=args.top_k)
        context = build_context(item.query, hits)
        if args.skip_generation:
            answer_text = item.reference_answer
            answer_meta = {"fallback": False, "model": "reference-answer-dry-run", "confidence": context.confidence}
        else:
            answer = generate_answer(item.query, context)
            answer_text = answer.answer
            answer_meta = {"fallback": answer.fallback, "model": answer.model, "confidence": answer.confidence}
        candidates.append(answer_text)
        references.append(item.reference_answer)
        rows.append(
            {
                "id": item.id,
                "category": item.category,
                "query": item.query,
                "reference_answer": item.reference_answer,
                "generated_answer": answer_text,
                "answer_meta": answer_meta,
                "retrieval": {
                    "top_k": args.top_k,
                    "hit_count": len(hits),
                    "top_citations": list(context.citations.values()),
                    "confidence": context.confidence,
                },
                "context_text": context.context_text,
                "rag_triad": compute_rag_triad(item.query, answer_text, context.context_text, item.reference_answer),
                "expected_checks": compute_expected_checks(
                    answer_text, context.context_text, item.expected_terms, item.expected_citations
                ),
            }
        )

    bertscore = compute_bertscore(candidates, references, model_type=args.bertscore_model)
    if bertscore.get("available"):
        for row, score_item in zip(rows, bertscore["items"]):
            row["bertscore"] = score_item
    else:
        for row in rows:
            row["bertscore"] = {"precision": None, "recall": None, "f1": None}

    summary = summarize(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "eval_overview_results.jsonl", rows)
    (out_dir / "eval_overview_summary.json").write_text(
        json.dumps({"summary": summary, "bertscore": bertscore}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(report_dir / "eval_overview.md", summary, rows, bertscore)
    print(json.dumps({"summary": summary, "bertscore_available": bertscore.get("available")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

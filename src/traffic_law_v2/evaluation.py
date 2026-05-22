from __future__ import annotations

from pathlib import Path
from typing import List

from traffic_law_v2.context import build_context
from traffic_law_v2.models import EvalCase
from traffic_law_v2.retrieval import HybridRetriever


DEFAULT_CASES: List[EvalCase] = [
    EvalCase("Đi ngược chiều trên đường một chiều bị phạt thế nào?", ("ngược chiều", "một chiều")),
    EvalCase("Chở quá số người quy định trên xe máy bị phạt bao nhiêu?", ("chở quá", "xe máy")),
    EvalCase("Không đội mũ bảo hiểm khi đi xe máy bị xử phạt ra sao?", ("mũ bảo hiểm",)),
    EvalCase("Vượt đèn đỏ bị phạt mức nào?", ("đèn đỏ",)),
    EvalCase("Điều kiện để thi giấy phép lái xe hạng A1 là gì?", ("A1", "giấy phép lái xe")),
]


def run_retrieval_eval(index_dir: Path, cases: List[EvalCase] | None = None) -> dict:
    retriever = HybridRetriever(index_dir)
    rows = []
    passed = 0
    for case in cases or DEFAULT_CASES:
        hits = retriever.retrieve(case.query, top_k=5)
        context = build_context(case.query, hits)
        haystack = context.context_text.lower()
        matched = [term for term in case.expected_terms if term.lower() in haystack]
        ok = bool(hits) and len(matched) >= max(1, len(case.expected_terms) // 2)
        passed += int(ok)
        rows.append(
            {
                "query": case.query,
                "ok": ok,
                "matched_terms": matched,
                "top_chunk": hits[0].chunk.chunk_id if hits else None,
                "confidence": context.confidence,
            }
        )
    total = len(rows)
    return {"total": total, "passed": passed, "pass_rate": round(passed / max(1, total), 4), "cases": rows}

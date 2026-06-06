from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(path: tuple[str, ...]) -> float | None:
        values = []
        for row in rows:
            cur: Any = row
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    cur = None
                    break
                cur = cur[key]
            if isinstance(cur, (int, float)):
                values.append(float(cur))
        return round(mean(values), 4) if values else None

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category") or "general")].append(row)

    return {
        "total_cases": len(rows),
        "averages": {
            "rag_context_relevance": avg(("rag_triad", "context_relevance")),
            "rag_groundedness": avg(("rag_triad", "groundedness")),
            "rag_answer_relevance": avg(("rag_triad", "answer_relevance")),
            "rag_triad_mean": avg(("rag_triad", "triad_mean")),
            "bertscore_precision": avg(("bertscore", "precision")),
            "bertscore_recall": avg(("bertscore", "recall")),
            "bertscore_f1": avg(("bertscore", "f1")),
            "answer_expected_term_coverage": avg(("expected_checks", "answer_expected_terms", "score")),
            "context_expected_term_coverage": avg(("expected_checks", "context_expected_terms", "score")),
        },
        "by_category": {
            category: {
                "count": len(items),
                "rag_triad_mean": round(mean(row["rag_triad"]["triad_mean"] for row in items), 4),
                "answer_expected_term_coverage": round(
                    mean(row["expected_checks"]["answer_expected_terms"]["score"] for row in items), 4
                ),
            }
            for category, items in sorted(by_category.items())
        },
    }


def write_markdown_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], bertscore_meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Eval Overview", ""]
    lines.append(f"Total cases: **{summary['total_cases']}**")
    lines.append("")
    lines.append("## BERTScore")
    if bertscore_meta.get("available"):
        lines.append(f"Model: `{bertscore_meta.get('model_type')}`")
    else:
        lines.append("BERTScore is not available in this environment.")
        lines.append(f"Error: `{bertscore_meta.get('error')}`")
    lines.append("")
    lines.append("## Average Metrics")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for key, value in summary["averages"].items():
        lines.append(f"| {key} | {value if value is not None else 'N/A'} |")
    lines.append("")
    lines.append("## Category Breakdown")
    lines.append("| Category | Count | RAG triad mean | Answer term coverage |")
    lines.append("| --- | ---: | ---: | ---: |")
    for category, item in summary["by_category"].items():
        lines.append(
            f"| {category} | {item['count']} | {item['rag_triad_mean']} | {item['answer_expected_term_coverage']} |"
        )
    lines.append("")
    lines.append("## Lowest RAG Triad Cases")
    for row in sorted(rows, key=lambda r: r["rag_triad"]["triad_mean"])[:5]:
        lines.append(f"- `{row['id']}` ({row['category']}): {row['rag_triad']['triad_mean']} - {row['query']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

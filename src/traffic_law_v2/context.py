from __future__ import annotations

from typing import Dict, List

from traffic_law_v2.models import ContextPackage, RetrievalHit
from traffic_law_v2.text import tokenize


def build_context(query: str, hits: List[RetrievalHit], max_tokens: int = 1800) -> ContextPackage:
    lines: List[str] = []
    citations: Dict[str, Dict[str, object]] = {}
    selected: List[RetrievalHit] = []
    used = 0
    for idx, hit in enumerate(hits, start=1):
        tokens = len(tokenize(hit.chunk.text))
        if selected and used + tokens > max_tokens:
            continue
        slot = f"C{len(selected) + 1}"
        meta = hit.chunk.metadata
        citations[slot] = {
            "chunk_id": hit.chunk.chunk_id,
            "document_title": meta.document_title,
            "article": meta.article,
            "chapter": meta.chapter,
            "section": meta.section,
            "page": meta.page,
            "table_id": meta.table_id,
            "source_path": meta.source_path,
            "score": round(hit.fused_score, 6),
        }
        header = citation_label(slot, citations[slot])
        lines.append(f"{header}\n{hit.chunk.text}")
        selected.append(hit)
        used += tokens
    confidence = min(0.95, 0.25 + sum(h.fused_score for h in selected[:3]) / 3.0) if selected else 0.0
    return ContextPackage(
        query=query,
        context_text="\n\n".join(lines),
        citations=citations,
        confidence=round(confidence, 4),
        hits=selected,
    )


def citation_label(slot: str, citation: Dict[str, object]) -> str:
    parts = [f"[{slot}]", str(citation.get("document_title") or "Tài liệu")]
    for key in ("article", "chapter", "page", "table_id"):
        value = citation.get(key)
        if value:
            parts.append(str(value))
    return " | ".join(parts)


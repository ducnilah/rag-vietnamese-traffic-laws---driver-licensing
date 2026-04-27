import re
from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple

from traffic_rag.offline.bm25 import tokenize
from traffic_rag.online.citation import build_citation
from traffic_rag.online.store import ChunkRecord, ChunkStore

CHUNK_INDEX_IN_ID_RE = re.compile(r"-(?:c|t)(\d+)-")


class ScoredHit(Protocol):
    chunk_id: str
    doc_id: str
    content_type: str
    text: str
    metadata: Dict[str, str]
    citation: str
    final_score: float


@dataclass(frozen=True)
class ContextChunk:
    slot: str
    chunk_id: str
    doc_id: str
    content_type: str
    citation: str
    text: str
    final_score: float


@dataclass(frozen=True)
class ContextPackage:
    query: str
    rewritten_query: str
    estimated_tokens: int
    confidence: float
    chunks: List[ContextChunk]
    citation_map: Dict[str, Dict[str, object]]
    context_text: str


def estimate_tokens(text: str) -> int:
    return len(tokenize(text))


def simple_query_rewrite(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip())


class ContextBuilder:
    def __init__(self, store: ChunkStore) -> None:
        self.store = store
        self._rows_by_doc: Dict[str, List[ChunkRecord]] = {}
        self._index_in_doc: Dict[str, Tuple[str, int]] = {}

        grouped: Dict[str, List[ChunkRecord]] = {}
        for row in store.all():
            grouped.setdefault(row.doc_id, []).append(row)

        for doc_id, rows in grouped.items():
            ordered = sorted(rows, key=self._chunk_order_key)
            self._rows_by_doc[doc_id] = ordered
            for idx, row in enumerate(ordered):
                self._index_in_doc[row.chunk_id] = (doc_id, idx)

    def build(
        self,
        query: str,
        hits: List[ScoredHit],
        neighbor_window: int = 1,
        max_context_tokens: int = 1800,
    ) -> ContextPackage:
        rewritten_query = simple_query_rewrite(query)
        expanded = self._expand_neighbors(hits, window=neighbor_window)
        selected = self._pack_to_budget(expanded, max_context_tokens=max_context_tokens)

        chunks: List[ContextChunk] = []
        citation_map: Dict[str, Dict[str, object]] = {}
        lines: List[str] = []

        for idx, hit in enumerate(selected):
            slot = f"C{idx + 1}"
            citation = hit.citation if hit.citation else build_citation(hit.metadata)
            chunk = ContextChunk(
                slot=slot,
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                content_type=hit.content_type,
                citation=citation,
                text=hit.text,
                final_score=hit.final_score,
            )
            chunks.append(chunk)
            citation_map[slot] = {
                "chunk_id": hit.chunk_id,
                "doc_id": hit.doc_id,
                "content_type": hit.content_type,
                "citation": citation,
                "score": round(hit.final_score, 6),
            }
            lines.append(f"[{slot}] {citation}\n{hit.text}".strip())

        context_text = "\n\n".join(lines).strip()
        estimated = sum(estimate_tokens(item.text) for item in chunks)
        confidence = self._compute_confidence(chunks)

        return ContextPackage(
            query=query,
            rewritten_query=rewritten_query,
            estimated_tokens=estimated,
            confidence=confidence,
            chunks=chunks,
            citation_map=citation_map,
            context_text=context_text,
        )

    @staticmethod
    def _chunk_order_key(row: ChunkRecord) -> Tuple[int, int]:
        chunk_index = row.metadata.get("chunk_index")
        if chunk_index is not None and str(chunk_index).isdigit():
            return (int(str(chunk_index)), row.start_char)
        match = CHUNK_INDEX_IN_ID_RE.search(row.chunk_id)
        if match:
            return (int(match.group(1)), row.start_char)
        return (10**9, row.start_char)

    def _expand_neighbors(self, hits: List[ScoredHit], window: int) -> List[ScoredHit]:
        if window <= 0 or not hits:
            return hits

        expanded: List[ScoredHit] = []
        seen = set()

        for hit in hits:
            if hit.chunk_id not in seen:
                expanded.append(hit)
                seen.add(hit.chunk_id)

            # Keep table hits as-is; neighbor expansion is more useful for text continuity.
            if hit.content_type == "table":
                continue

            loc = self._index_in_doc.get(hit.chunk_id)
            if not loc:
                continue
            doc_id, center = loc
            rows = self._rows_by_doc.get(doc_id, [])
            start = max(0, center - window)
            end = min(len(rows), center + window + 1)

            for idx in range(start, end):
                row = rows[idx]
                if row.chunk_id in seen:
                    continue
                # Synthetic neighbor hit keeps base score slightly discounted.
                expanded.append(
                    _NeighborHit(
                        chunk_id=row.chunk_id,
                        doc_id=row.doc_id,
                        content_type=row.metadata.get("content_type", "text"),
                        text=row.text,
                        metadata=dict(row.metadata),
                        citation="",
                        final_score=max(0.0, hit.final_score * 0.9),
                    )
                )
                seen.add(row.chunk_id)

        return expanded

    @staticmethod
    def _pack_to_budget(hits: List[ScoredHit], max_context_tokens: int) -> List[ScoredHit]:
        selected: List[ScoredHit] = []
        used = 0
        for hit in hits:
            tokens = estimate_tokens(hit.text)
            if (used + tokens) > max_context_tokens:
                if not selected:
                    remaining = max(1, max_context_tokens - used)
                    truncated = ContextBuilder._truncate_hit(hit, token_limit=remaining)
                    selected.append(truncated)
                    used += estimate_tokens(truncated.text)
                continue
            selected.append(hit)
            used += tokens
            if used >= max_context_tokens:
                break
        return selected

    @staticmethod
    def _truncate_hit(hit: ScoredHit, token_limit: int) -> "_NeighborHit":
        raw_tokens = tokenize(hit.text)
        limited_text = " ".join(raw_tokens[:token_limit]).strip()
        if not limited_text:
            limited_text = hit.text[:200]
        return _NeighborHit(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            content_type=hit.content_type,
            text=limited_text,
            metadata=dict(hit.metadata),
            citation=hit.citation,
            final_score=hit.final_score,
        )

    @staticmethod
    def _compute_confidence(chunks: List[ContextChunk]) -> float:
        if not chunks:
            return 0.0

        top_scores = sorted((max(0.0, item.final_score) for item in chunks), reverse=True)[:3]
        max_score = max(top_scores) if top_scores else 0.0
        score_signal = (
            (sum(score / max_score for score in top_scores) / len(top_scores))
            if max_score > 0
            else 0.0
        )
        citation_signal = 1.0 if all(item.citation for item in chunks) else 0.7
        doc_diversity = len({item.doc_id for item in chunks}) / len(chunks)
        confidence = (0.55 * score_signal) + (0.25 * citation_signal) + (0.20 * doc_diversity)
        return round(max(0.0, min(1.0, confidence)), 4)


@dataclass(frozen=True)
class _NeighborHit:
    chunk_id: str
    doc_id: str
    content_type: str
    text: str
    metadata: Dict[str, str]
    citation: str
    final_score: float

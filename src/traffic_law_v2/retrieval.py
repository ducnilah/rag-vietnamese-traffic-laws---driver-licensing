from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List

from traffic_law_v2.config import get_settings
from traffic_law_v2.embeddings import HashEmbedding, cosine, get_embeddings
from traffic_law_v2.indexing import SimpleBM25, load_chunks
from traffic_law_v2.models import Chunk, RetrievalHit
from traffic_law_v2.vectorstore import search_chroma


class HybridRetriever:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.chunks = load_chunks(index_dir)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = SimpleBM25(self.chunks)
        self.embeddings = get_embeddings(get_settings())
        # Do not eagerly embed the full corpus on every request.
        # In API-backed mode this is expensive and can fail even though Chroma
        # already stores persisted document vectors.
        self.chunk_vectors = None

    def retrieve(self, query: str, top_k: int = 6, candidate_k: int = 24) -> List[RetrievalHit]:
        sparse = dict(self.bm25.search(query, top_k=candidate_k))
        dense = self._dense_search(query, top_k=candidate_k)
        dense_scores = dict(dense)
        ids = list(dict.fromkeys([cid for cid, _ in dense] + list(sparse.keys())))
        sparse_norm = _normalize(sparse)
        dense_norm = _normalize(dense_scores)
        hits: List[RetrievalHit] = []
        for cid in ids:
            chunk = self.by_id.get(cid)
            if not chunk:
                continue
            fused = 0.52 * sparse_norm.get(cid, 0.0) + 0.48 * dense_norm.get(cid, 0.0)
            if _has_table_intent(query) and ("[TABLE" in chunk.text or chunk.metadata.table_id):
                fused *= 1.25
            fused *= _vehicle_profile_boost(query, chunk)
            fused *= _query_topic_boost(query, chunk)
            hits.append(
                RetrievalHit(
                    chunk=chunk,
                    sparse_score=sparse.get(cid, 0.0),
                    dense_score=dense_scores.get(cid, 0.0),
                    fused_score=fused,
                )
            )
        deduped = _dedupe(hits)
        return sorted(deduped, key=lambda h: h.fused_score, reverse=True)[:top_k]

    def _dense_search(self, query: str, top_k: int) -> List[tuple[str, float]]:
        try:
            return search_chroma(self.index_dir, query, top_k=top_k)
        except Exception:
            # If API-backed embeddings fail at query time, keep the app alive
            # and let sparse retrieval carry the request instead of crashing.
            if not isinstance(self.embeddings, HashEmbedding):
                return []
        if self.chunk_vectors is None:
            self.chunk_vectors = self.embeddings.embed_documents([c.text for c in self.chunks]) if self.chunks else []
        qv = self.embeddings.embed_query(query)
        scored = [(chunk.chunk_id, cosine(qv, vec)) for chunk, vec in zip(self.chunks, self.chunk_vectors)]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _dedupe(hits: List[RetrievalHit]) -> List[RetrievalHit]:
    out: List[RetrievalHit] = []
    seen = set()
    for hit in sorted(hits, key=lambda h: h.fused_score, reverse=True):
        fingerprint = " ".join(hit.chunk.text.lower().split()[:60])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(hit)
    return out


def _has_table_intent(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in ("bảng", "mức phạt", "bao nhiêu", "điểm", "tiền phạt"))


def _vehicle_profile_boost(query: str, chunk: Chunk) -> float:
    q = query.lower()
    article = (chunk.metadata.article or "").lower()
    text_head = chunk.text[:350].lower()
    haystack = f"{article} {text_head}"
    wants_motorbike = any(term in q for term in ("xe máy", "mô tô", "gắn máy", "a1"))
    wants_car = any(term in q for term in ("ô tô", "oto", "xe hơi", "b1", "b2", "xe khách"))
    if wants_motorbike:
        if "xe mô tô" in haystack or "xe gắn máy" in haystack or "điều 6" in article:
            return 1.22
        if "xe ô tô" in haystack or "ô tô chở" in haystack:
            return 0.82
    if wants_car:
        if "xe ô tô" in haystack or "ô tô chở" in haystack:
            return 1.16
        if "xe mô tô" in haystack or "xe gắn máy" in haystack:
            return 0.86
    return 1.0


def _query_topic_boost(query: str, chunk: Chunk) -> float:
    q = query.lower()
    article = (chunk.metadata.article or "").lower()
    title = (chunk.metadata.document_title or "").lower()
    text_head = chunk.text[:900].lower()
    haystack = f"{title} {article} {text_head}"

    score = 1.0
    terms = _query_terms(q)
    if terms:
        overlap = sum(1 for term in terms if term in haystack)
        score *= 1.0 + min(0.65, overlap / max(1, len(terms)) * 0.65)

    if ("định nghĩa" in q or "là gì" in q) and ("giải thích từ ngữ" in article or "giải thích từ ngữ" in haystack):
        score *= 1.35
    if ("lĩnh vực nào" in q or "phạm vi" in q) and ("phạm vi điều chỉnh" in article or "phạm vi điều chỉnh" in haystack):
        score *= 1.35
    if ("lĩnh vực nào" in q or "phạm vi" in q):
        if "điều 1" in article or "nghị định quy định xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ và đường sắt" in haystack:
            score *= 1.45
        if "thẩm quyền xử phạt" in article or "nguyên tắc xác định thẩm quyền" in article:
            score *= 0.65
    if "biện pháp khắc phục hậu quả" in q and "biện pháp khắc phục hậu quả" in haystack:
        score *= 1.35
    if "biện pháp khắc phục hậu quả" in q:
        if "điều 4" in article or "buộc phải tháo dỡ" in haystack or "buộc phải di dời" in haystack:
            score *= 1.6
        if "thẩm quyền xử phạt" in article:
            score *= 0.7

    if any(t in q for t in ("ô tô", "oto", "xe hơi")) and "người đi bộ" in article:
        score *= 0.55
    if any(t in q for t in ("ô tô", "oto", "xe hơi")) and "xe đạp" in article:
        score *= 0.60
    if any(t in q for t in ("xe máy", "mô tô", "gắn máy")) and "xe ô tô" in article:
        score *= 0.75

    if "nồng độ cồn" in q:
        if "nồng độ cồn" in haystack:
            score *= 1.4
        else:
            score *= 0.75
    if "đường cao tốc" in q and "đường cao tốc" not in haystack:
        score *= 0.75

    return max(0.25, min(score, 2.5))


def _query_terms(q: str) -> list[str]:
    raw = [t for t in re.findall(r"[\wÀ-ỹĐđ]+", q) if len(t) >= 3]
    stop = {
        "người",
        "điều",
        "khiển",
        "trong",
        "theo",
        "những",
        "như",
        "thế",
        "nào",
        "bao",
        "nhiêu",
        "một",
        "các",
        "đối",
        "với",
    }
    return [t for t in raw if t not in stop]

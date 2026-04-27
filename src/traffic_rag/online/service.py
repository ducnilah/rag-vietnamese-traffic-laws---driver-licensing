import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

from traffic_rag.offline.bm25 import BM25Index, tokenize
from traffic_rag.online.citation import build_citation
from traffic_rag.online.context import ContextBuilder, ContextPackage
from traffic_rag.online.dense import ChromaDenseRetriever, DenseRetriever, JaccardDenseRetriever
from traffic_rag.online.fusion import (
    minmax_normalize,
    reciprocal_rank_fusion,
    weighted_hybrid_score,
)
from traffic_rag.online.store import ChunkStore
from traffic_rag.vector.chroma import CHROMA_COLLECTION_DEFAULT, CHROMA_DIRNAME_DEFAULT

logger = logging.getLogger(__name__)

TABLE_INTENT_TERMS = {
    "bang",
    "bảng",
    "bieu",
    "biểu",
    "phu luc",
    "phụ lục",
    "muc",
    "mức",
    "so lieu",
    "số liệu",
    "chi tiet",
    "chi tiết",
    "cot",
    "cột",
    "hang",
    "hàng",
}


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def has_table_intent(query: str) -> bool:
    normalized = normalize_query(query)
    return any(term in normalized for term in TABLE_INTENT_TERMS)


@dataclass(frozen=True)
class HybridHit:
    chunk_id: str
    doc_id: str
    content_type: str
    text: str
    metadata: Dict[str, str]
    citation: str
    sparse_score: float
    dense_score: float
    fused_score: float
    final_score: float


class RetrievalService:
    def __init__(
        self,
        index_dir: Path,
        dense_retriever: Optional[DenseRetriever] = None,
        dense_backend: Literal["auto", "chroma", "jaccard"] = "auto",
        chroma_dir: Optional[Path] = None,
        chroma_collection: str = CHROMA_COLLECTION_DEFAULT,
    ) -> None:
        self.index_dir = index_dir
        self.bm25 = BM25Index.load(index_dir / "bm25.json")
        self.store = ChunkStore.load(index_dir / "chunks.jsonl")
        self.context_builder = ContextBuilder(self.store)
        self.dense_backend = dense_backend
        if dense_retriever:
            self.dense = dense_retriever
            self.dense_backend_name = dense_retriever.__class__.__name__
        else:
            self.dense, self.dense_backend_name = self._resolve_dense_backend(
                dense_backend=dense_backend,
                chroma_dir=chroma_dir,
                chroma_collection=chroma_collection,
            )

    def _resolve_dense_backend(
        self,
        dense_backend: Literal["auto", "chroma", "jaccard"],
        chroma_dir: Optional[Path],
        chroma_collection: str,
    ) -> tuple[DenseRetriever, str]:
        if dense_backend in {"auto", "chroma"}:
            vector_dir = chroma_dir if chroma_dir else self.index_dir / CHROMA_DIRNAME_DEFAULT
            chroma = ChromaDenseRetriever.try_create(
                chroma_dir=vector_dir,
                collection_name=chroma_collection,
            )
            if chroma is not None:
                logger.info(
                    "dense_backend_selected backend=chroma dir=%s collection=%s",
                    vector_dir,
                    chroma_collection,
                )
                return chroma, "chroma"
            if dense_backend == "chroma":
                raise RuntimeError(
                    "Dense backend is set to 'chroma' but collection is unavailable. "
                    "Rebuild index with --with-chroma and ensure chromadb is installed."
                )
        logger.info("dense_backend_selected backend=jaccard")
        return JaccardDenseRetriever(self.store), "jaccard"

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 30,
        use_hybrid: bool = True,
        table_boost: float = 1.35,
        apply_diversity: bool = True,
    ) -> List[HybridHit]:
        logger.info(
            "service.retrieve query=%r top_k=%d candidate_k=%d use_hybrid=%s",
            query,
            top_k,
            candidate_k,
            use_hybrid,
        )
        table_intent = has_table_intent(query)

        sparse_hits = self.bm25.search(query, top_k=max(top_k, candidate_k))
        sparse_scores: Dict[str, float] = {hit.chunk_id: hit.score for hit in sparse_hits}

        dense_scores: Dict[str, float] = {}
        if use_hybrid:
            dense_hits = self.dense.search(query, top_k=max(top_k, candidate_k))
            dense_scores = {hit.chunk_id: hit.score for hit in dense_hits}

        candidate_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
        sparse_norm = minmax_normalize({k: sparse_scores.get(k, 0.0) for k in candidate_ids})
        dense_norm = minmax_normalize({k: dense_scores.get(k, 0.0) for k in candidate_ids})

        sparse_rank = {hit.chunk_id: i + 1 for i, hit in enumerate(sparse_hits)}
        dense_rank: Dict[str, int] = {}
        if use_hybrid:
            dense_rank = {
                cid: i + 1
                for i, cid in enumerate(
                    sorted(dense_scores.keys(), key=lambda key: dense_scores[key], reverse=True)
                )
            }
        rrf_scores = reciprocal_rank_fusion([sparse_rank, dense_rank] if use_hybrid else [sparse_rank])
        rrf_norm = minmax_normalize(rrf_scores)

        results: List[HybridHit] = []
        for chunk_id in candidate_ids:
            row = self.store.get(chunk_id)
            metadata = dict(row.metadata)
            content_type = metadata.get("content_type", "text")
            sparse_score = sparse_scores.get(chunk_id, 0.0)
            dense_score = dense_scores.get(chunk_id, 0.0)
            fused_score = weighted_hybrid_score(
                sparse_norm=sparse_norm.get(chunk_id, 0.0),
                dense_norm=dense_norm.get(chunk_id, 0.0),
                rrf_score=rrf_norm.get(chunk_id, 0.0),
            )
            final_score = fused_score * table_boost if (table_intent and content_type == "table") else fused_score

            results.append(
                HybridHit(
                    chunk_id=chunk_id,
                    doc_id=row.doc_id,
                    content_type=content_type,
                    text=row.text,
                    metadata=metadata,
                    citation=build_citation(metadata),
                    sparse_score=sparse_score,
                    dense_score=dense_score,
                    fused_score=fused_score,
                    final_score=final_score,
                )
            )

        results.sort(key=lambda item: item.final_score, reverse=True)

        if apply_diversity and len(results) > 1:
            mmr_pool = results[: max(candidate_k, top_k * 6)]
            reranked = self._mmr_rerank(mmr_pool, lambda_mult=0.75)
            deduped = self._suppress_near_duplicates(reranked, top_k=top_k, threshold=0.85)
            logger.info(
                "service.post_rank total=%d pool=%d deduped=%d",
                len(results),
                len(mmr_pool),
                len(deduped),
            )
            return deduped
        return results[:top_k]

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 30,
        use_hybrid: bool = True,
        table_boost: float = 1.35,
        apply_diversity: bool = True,
        neighbor_window: int = 1,
        max_context_tokens: int = 1800,
    ) -> ContextPackage:
        hits = self.retrieve(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
            use_hybrid=use_hybrid,
            table_boost=table_boost,
            apply_diversity=apply_diversity,
        )
        return self.context_builder.build(
            query=query,
            hits=hits,
            neighbor_window=neighbor_window,
            max_context_tokens=max_context_tokens,
        )

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(tokenize(text))

    @classmethod
    def _text_similarity(cls, a: str, b: str) -> float:
        a_tokens = cls._token_set(a)
        b_tokens = cls._token_set(b)
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens.intersection(b_tokens))
        if inter == 0:
            return 0.0
        union = len(a_tokens.union(b_tokens))
        return inter / float(union)

    @classmethod
    def _mmr_rerank(cls, hits: List[HybridHit], lambda_mult: float = 0.75) -> List[HybridHit]:
        if len(hits) <= 1:
            return hits

        max_score = max(hit.final_score for hit in hits) or 1.0
        relevance = {hit.chunk_id: hit.final_score / max_score for hit in hits}

        selected: List[HybridHit] = []
        remaining = hits[:]
        while remaining:
            best: Optional[HybridHit] = None
            best_value = float("-inf")
            for candidate in remaining:
                if not selected:
                    diversity_penalty = 0.0
                else:
                    diversity_penalty = max(
                        cls._text_similarity(candidate.text, chosen.text) for chosen in selected
                    )
                value = (lambda_mult * relevance[candidate.chunk_id]) - (
                    (1.0 - lambda_mult) * diversity_penalty
                )
                if value > best_value:
                    best_value = value
                    best = candidate
            assert best is not None
            selected.append(best)
            remaining = [item for item in remaining if item.chunk_id != best.chunk_id]
        return selected

    @classmethod
    def _suppress_near_duplicates(
        cls,
        hits: List[HybridHit],
        top_k: int,
        threshold: float = 0.85,
    ) -> List[HybridHit]:
        selected: List[HybridHit] = []
        for hit in hits:
            is_duplicate = False
            for chosen in selected:
                if hit.content_type != chosen.content_type:
                    continue
                if hit.doc_id != chosen.doc_id:
                    continue
                similarity = cls._text_similarity(hit.text, chosen.text)
                if similarity >= threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                selected.append(hit)
            if len(selected) >= top_k:
                break
        return selected[:top_k]

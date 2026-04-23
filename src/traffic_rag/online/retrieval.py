import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from traffic_rag.offline.bm25 import BM25Index

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


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def has_table_intent(query: str) -> bool:
    normalized = _normalize_query(query)
    for term in TABLE_INTENT_TERMS:
        if term in normalized:
            return True
    return False


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    raw_score: float
    final_score: float
    content_type: str
    text: str
    metadata: Dict[str, str]


def _load_chunk_map(chunks_path: Path) -> Dict[str, Dict[str, object]]:
    chunk_map: Dict[str, Dict[str, object]] = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunk_map[str(row["chunk_id"])] = row
    return chunk_map


def search_with_table_priority(
    index_dir: Path,
    query: str,
    top_k: int = 5,
    candidate_k: int = 30,
    table_boost: float = 1.35,
) -> List[RetrievalHit]:
    bm25 = BM25Index.load(index_dir / "bm25.json")
    chunk_map = _load_chunk_map(index_dir / "chunks.jsonl")
    table_intent = has_table_intent(query)
    logger.info(
        "retrieval.start query=%r table_intent=%s top_k=%d candidate_k=%d",
        query,
        table_intent,
        top_k,
        candidate_k,
    )

    candidates = bm25.search(query, top_k=max(top_k, candidate_k))
    logger.debug("retrieval.candidates=%d", len(candidates))
    rescored: List[RetrievalHit] = []
    for hit in candidates:
        row = chunk_map.get(hit.chunk_id)
        if row is None:
            continue
        metadata = dict(row.get("metadata", {}))
        content_type = metadata.get("content_type", "text")
        factor = table_boost if (table_intent and content_type == "table") else 1.0
        rescored.append(
            RetrievalHit(
                chunk_id=hit.chunk_id,
                raw_score=hit.score,
                final_score=hit.score * factor,
                content_type=content_type,
                text=str(row.get("text", "")),
                metadata=metadata,
            )
        )

    rescored.sort(key=lambda item: item.final_score, reverse=True)
    top_types = [item.content_type for item in rescored[:top_k]]
    logger.info("retrieval.done hits=%d top_types=%s", min(top_k, len(rescored)), top_types)
    return rescored[:top_k]

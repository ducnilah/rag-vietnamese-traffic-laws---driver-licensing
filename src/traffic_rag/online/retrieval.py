import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

from traffic_rag.online.service import (
    HybridHit,
    RetrievalService,
    has_table_intent,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    raw_score: float
    final_score: float
    content_type: str
    text: str
    metadata: Dict[str, str]


def search_with_table_priority(
    index_dir: Path,
    query: str,
    top_k: int = 5,
    candidate_k: int = 30,
    table_boost: float = 1.35,
) -> List[RetrievalHit]:
    service = RetrievalService(index_dir, dense_backend="jaccard")
    hits = service.retrieve(
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        use_hybrid=False,
        table_boost=table_boost,
    )
    return [
        RetrievalHit(
            chunk_id=hit.chunk_id,
            raw_score=hit.sparse_score,
            final_score=hit.final_score,
            content_type=hit.content_type,
            text=hit.text,
            metadata=hit.metadata,
        )
        for hit in hits
    ]


def search_hybrid(
    index_dir: Path,
    query: str,
    top_k: int = 5,
    candidate_k: int = 30,
    table_boost: float = 1.35,
    dense_backend: Literal["auto", "chroma", "jaccard"] = "auto",
) -> List[HybridHit]:
    service = RetrievalService(index_dir, dense_backend=dense_backend)
    return service.retrieve(
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
        use_hybrid=True,
        table_boost=table_boost,
    )

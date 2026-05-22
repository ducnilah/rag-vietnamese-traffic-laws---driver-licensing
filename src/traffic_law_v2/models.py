from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LegalMetadata(BaseModel):
    source_id: str
    source_path: str
    document_title: str = "Unknown document"
    legal_type: str = "unknown"
    issuing_body: Optional[str] = None
    effective_date: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    clause: Optional[str] = None
    point: Optional[str] = None
    page: Optional[str] = None
    table_id: Optional[str] = None
    version: str = "v1"


class SourceDocument(BaseModel):
    doc_id: str
    text: str
    metadata: LegalMetadata


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: LegalMetadata
    chunk_index: int
    token_count: int


class RetrievalHit(BaseModel):
    chunk: Chunk
    sparse_score: float = 0.0
    dense_score: float = 0.0
    fused_score: float = 0.0


class ContextPackage(BaseModel):
    query: str
    context_text: str
    citations: Dict[str, Dict[str, Any]]
    confidence: float
    hits: List[RetrievalHit]


class AnswerPackage(BaseModel):
    answer: str
    citations: Dict[str, Dict[str, Any]]
    confidence: float
    model: str
    trace_id: str
    fallback: bool = False


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_terms: tuple[str, ...] = ()

from pathlib import Path
from typing import Any, Dict

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = None  # type: ignore
    FASTAPI_AVAILABLE = False

from traffic_rag.online.service import RetrievalService, has_table_intent


if FASTAPI_AVAILABLE:
    class RetrieveRequest(BaseModel):  # type: ignore[misc]
        query: str
        top_k: int = Field(default=5, ge=1, le=50)
        candidate_k: int = Field(default=30, ge=1, le=200)
        mode: str = Field(default="hybrid", pattern="^(hybrid|sparse)$")
        dense_backend: str = Field(default="auto", pattern="^(auto|chroma|jaccard)$")

    class ContextRequest(BaseModel):  # type: ignore[misc]
        query: str
        top_k: int = Field(default=5, ge=1, le=50)
        candidate_k: int = Field(default=30, ge=1, le=200)
        neighbor_window: int = Field(default=1, ge=0, le=5)
        max_context_tokens: int = Field(default=1800, ge=64, le=12000)
        mode: str = Field(default="hybrid", pattern="^(hybrid|sparse)$")
        dense_backend: str = Field(default="auto", pattern="^(auto|chroma|jaccard)$")
else:
    class RetrieveRequest:  # pragma: no cover
        pass

    class ContextRequest:  # pragma: no cover
        pass


def create_app(index_dir: Path) -> "FastAPI":
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "fastapi/pydantic is not installed. Install with: python3 -m pip install fastapi uvicorn pydantic"
        )

    service = RetrievalService(index_dir)
    app = FastAPI(title="Traffic RAG Retrieval API", version="0.1.0")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "index_dir": str(index_dir),
            "total_chunks": len(service.store.all()),
        }

    @app.post("/retrieve")
    def retrieve(req: RetrieveRequest) -> Dict[str, Any]:
        use_hybrid = req.mode == "hybrid"
        if use_hybrid:
            service_local = RetrievalService(index_dir, dense_backend=req.dense_backend)
        else:
            service_local = RetrievalService(index_dir, dense_backend="jaccard")
        hits = service_local.retrieve(
            query=req.query,
            top_k=req.top_k,
            candidate_k=req.candidate_k,
            use_hybrid=use_hybrid,
        )
        return {
            "query": req.query,
            "mode": req.mode,
            "dense_backend": req.dense_backend if use_hybrid else None,
            "table_intent": has_table_intent(req.query),
            "hits": [
                {
                    "chunk_id": hit.chunk_id,
                    "content_type": hit.content_type,
                    "sparse_score": round(hit.sparse_score, 6),
                    "dense_score": round(hit.dense_score, 6),
                    "fused_score": round(hit.fused_score, 6),
                    "final_score": round(hit.final_score, 6),
                    "table_id": hit.metadata.get("table_id"),
                    "page": hit.metadata.get("page"),
                    "citation": hit.citation,
                    "preview": hit.text[:300],
                }
                for hit in hits
            ],
        }

    @app.post("/context")
    def context(req: ContextRequest) -> Dict[str, Any]:
        use_hybrid = req.mode == "hybrid"
        if use_hybrid:
            service_local = RetrievalService(index_dir, dense_backend=req.dense_backend)
        else:
            service_local = RetrievalService(index_dir, dense_backend="jaccard")

        package = service_local.build_context(
            query=req.query,
            top_k=req.top_k,
            candidate_k=req.candidate_k,
            use_hybrid=use_hybrid,
            neighbor_window=req.neighbor_window,
            max_context_tokens=req.max_context_tokens,
        )
        return {
            "query": package.query,
            "rewritten_query": package.rewritten_query,
            "mode": req.mode,
            "dense_backend": req.dense_backend if use_hybrid else None,
            "estimated_tokens": package.estimated_tokens,
            "confidence": package.confidence,
            "citation_map": package.citation_map,
            "chunks": [
                {
                    "slot": chunk.slot,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "content_type": chunk.content_type,
                    "score": round(chunk.final_score, 6),
                    "citation": chunk.citation,
                    "preview": chunk.text[:300],
                }
                for chunk in package.chunks
            ],
            "context_text": package.context_text,
        }

    return app

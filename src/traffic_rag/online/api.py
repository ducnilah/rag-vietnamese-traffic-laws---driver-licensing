from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI
    from fastapi import HTTPException
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = RuntimeError  # type: ignore
    BaseModel = object  # type: ignore
    Field = None  # type: ignore
    FASTAPI_AVAILABLE = False

from traffic_rag.online.generator import ChatGenerator
from traffic_rag.online.guardrails import evaluate_query_guardrails
from traffic_rag.online.service import RetrievalService, has_table_intent
from traffic_rag.state.service import ConversationService, SQLALCHEMY_AVAILABLE


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

    class ThreadCreateRequest(BaseModel):  # type: ignore[misc]
        user_id: str
        title: str = "New chat"

    class MessageCreateRequest(BaseModel):  # type: ignore[misc]
        role: str = Field(pattern="^(user|assistant|system)$")
        content: str
        citations: Optional[Dict[str, Any]] = None

    class MemoryItemRequest(BaseModel):  # type: ignore[misc]
        key: str
        value: str
        confidence: float = Field(default=0.7, ge=0.0, le=1.0)

    class MemoryPatchRequest(BaseModel):  # type: ignore[misc]
        items: List[MemoryItemRequest]

    class ChatRequest(BaseModel):  # type: ignore[misc]
        user_id: str
        query: str
        mode: str = Field(default="hybrid", pattern="^(hybrid|sparse)$")
        dense_backend: str = Field(default="auto", pattern="^(auto|chroma|jaccard)$")
        top_k: int = Field(default=5, ge=1, le=50)
        candidate_k: int = Field(default=30, ge=1, le=200)
        neighbor_window: int = Field(default=1, ge=0, le=5)
        max_context_tokens: int = Field(default=1800, ge=64, le=12000)
else:
    class RetrieveRequest:  # pragma: no cover
        pass

    class ContextRequest:  # pragma: no cover
        pass

    class ThreadCreateRequest:  # pragma: no cover
        pass

    class MessageCreateRequest:  # pragma: no cover
        pass

    class MemoryPatchRequest:  # pragma: no cover
        pass

    class ChatRequest:  # pragma: no cover
        pass


def create_app(index_dir: Path, db_url: str = "sqlite:///data/app.db") -> "FastAPI":
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "fastapi/pydantic is not installed. Install with: python3 -m pip install fastapi uvicorn pydantic"
        )

    service = RetrievalService(index_dir)
    generator = ChatGenerator.from_env()
    conversation = ConversationService(db_url) if SQLALCHEMY_AVAILABLE else None
    if conversation:
        conversation.create_schema()
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

    @app.post("/threads")
    def create_thread(req: ThreadCreateRequest) -> Dict[str, Any]:
        if conversation is None:
            raise RuntimeError("sqlalchemy not installed for conversation endpoints")
        thread = conversation.create_thread(req.user_id, req.title)
        return thread.__dict__

    @app.get("/threads")
    def list_threads(user_id: str, limit: int = 20) -> Dict[str, Any]:
        if conversation is None:
            raise RuntimeError("sqlalchemy not installed for conversation endpoints")
        rows = conversation.list_threads(user_id, limit=limit)
        return {"threads": [row.__dict__ for row in rows]}

    @app.get("/threads/{thread_id}")
    def get_thread(thread_id: str) -> Dict[str, Any]:
        if conversation is None:
            raise RuntimeError("sqlalchemy not installed for conversation endpoints")
        row = conversation.get_thread(thread_id)
        if row is None:
            return {"thread": None}
        return {"thread": row.__dict__}

    @app.post("/threads/{thread_id}/messages")
    def add_message(thread_id: str, req: MessageCreateRequest) -> Dict[str, Any]:
        if conversation is None:
            raise RuntimeError("sqlalchemy not installed for conversation endpoints")
        msg = conversation.add_message(thread_id, req.role, req.content, req.citations)
        return msg.__dict__

    @app.get("/threads/{thread_id}/messages")
    def list_messages(thread_id: str, limit: int = 50) -> Dict[str, Any]:
        if conversation is None:
            raise RuntimeError("sqlalchemy not installed for conversation endpoints")
        rows = conversation.list_messages(thread_id, limit=limit)
        return {"messages": [row.__dict__ for row in rows]}

    @app.post("/threads/{thread_id}/summary:refresh")
    def refresh_summary(thread_id: str) -> Dict[str, Any]:
        if conversation is None:
            raise RuntimeError("sqlalchemy not installed for conversation endpoints")
        summary = conversation.refresh_summary(thread_id)
        return {"thread_id": thread_id, "summary": summary}

    @app.get("/threads/{thread_id}/summary")
    def get_summary(thread_id: str) -> Dict[str, Any]:
        if conversation is None:
            raise RuntimeError("sqlalchemy not installed for conversation endpoints")
        return {"thread_id": thread_id, "summary": conversation.get_summary(thread_id)}

    @app.patch("/users/{user_id}/memory")
    def patch_memory(user_id: str, req: MemoryPatchRequest) -> Dict[str, Any]:
        if conversation is None:
            raise RuntimeError("sqlalchemy not installed for conversation endpoints")
        for item in req.items:
            conversation.upsert_memory(
                user_id=user_id,
                key=item.key,
                value=item.value,
                confidence=item.confidence,
            )
        return {"ok": True, "items": conversation.list_memory(user_id)}

    @app.get("/users/{user_id}/memory")
    def get_memory(user_id: str) -> Dict[str, Any]:
        if conversation is None:
            raise RuntimeError("sqlalchemy not installed for conversation endpoints")
        return {"items": conversation.list_memory(user_id)}

    @app.post("/threads/{thread_id}/chat")
    def chat(thread_id: str, req: ChatRequest) -> Dict[str, Any]:
        if conversation is None:
            raise RuntimeError("sqlalchemy not installed for conversation endpoints")

        thread = conversation.get_thread(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="thread not found")
        if thread.user_id != req.user_id:
            raise HTTPException(status_code=403, detail="thread does not belong to user")

        guard = evaluate_query_guardrails(req.query)
        conversation.add_message(thread_id, "user", req.query)

        if not guard.allow:
            msg = conversation.add_message(
                thread_id,
                "assistant",
                guard.message,
                citations={"guardrail": {"code": guard.code, "risks": guard.risks}},
            )
            return {
                "thread_id": thread_id,
                "guardrail": {"allow": guard.allow, "code": guard.code, "risks": guard.risks},
                "assistant_message": msg.__dict__,
                "context": None,
            }

        use_hybrid = req.mode == "hybrid"
        if use_hybrid:
            service_local = RetrievalService(index_dir, dense_backend=req.dense_backend)
        else:
            service_local = RetrievalService(index_dir, dense_backend="jaccard")
        context = service_local.build_context(
            query=req.query,
            top_k=req.top_k,
            candidate_k=req.candidate_k,
            use_hybrid=use_hybrid,
            neighbor_window=req.neighbor_window,
            max_context_tokens=req.max_context_tokens,
        )
        generated = generator.generate(req.query, context)

        assistant = conversation.add_message(
            thread_id,
            "assistant",
            generated.answer,
            citations={
                "citation_map": context.citation_map,
                "confidence": context.confidence,
                "model": generated.model,
                "fallback": generated.used_fallback,
            },
        )
        return {
            "thread_id": thread_id,
            "guardrail": {"allow": True, "code": "OK", "risks": []},
            "assistant_message": assistant.__dict__,
            "context": {
                "rewritten_query": context.rewritten_query,
                "estimated_tokens": context.estimated_tokens,
                "confidence": context.confidence,
                "citation_map": context.citation_map,
            },
            "generation": {
                "model": generated.model,
                "fallback": generated.used_fallback,
            },
        }

    return app

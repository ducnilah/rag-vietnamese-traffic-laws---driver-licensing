from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from traffic_law_v2.context import build_context
from traffic_law_v2.config import get_settings
from traffic_law_v2.generation import generate_answer
from traffic_law_v2.indexing import build_index
from traffic_law_v2.retrieval import HybridRetriever
from traffic_law_v2.state import create_state, extract_memory_facts


def create_app() -> FastAPI:
    settings = get_settings()
    state = create_state(settings)
    app = FastAPI(
        title="Traffic Law V2",
        version="0.1.0",
        description="LangChain-first RAG assistant for Vietnamese traffic law.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get(f"{settings.api_prefix}/health")
    def health() -> dict:
        return {
            "ok": True,
            "env": settings.app_env,
            "provider": settings.model_provider,
            "llm_model": settings.llm_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "api_prefix": settings.api_prefix,
        }

    class BuildIndexRequest(BaseModel):
        raw_dir: str = "data/raw"
        index_dir: str = "data/index"

    class RetrieveRequest(BaseModel):
        query: str
        index_dir: str = "data/index"
        top_k: int = Field(default=5, ge=1, le=20)

    class ChatRequest(BaseModel):
        user_id: str = "dev-user"
        query: str
        index_dir: str = "data/index"
        top_k: int = Field(default=5, ge=1, le=20)

    class AuthRequest(BaseModel):
        email: str
        password: str = Field(min_length=6)

    class FeedbackRequest(BaseModel):
        user_id: str = "dev-user"
        message_id: str
        rating: int = Field(ge=-1, le=1)
        note: str = ""

    @app.post(f"{settings.api_prefix}/indexes/build")
    def build(req: BuildIndexRequest) -> dict:
        from pathlib import Path

        return build_index(Path(req.raw_dir), Path(req.index_dir))

    @app.post(f"{settings.api_prefix}/auth/register")
    def auth_register(req: AuthRequest) -> dict:
        try:
            user = state.register_user(req.email, req.password)
            _, token = state.login(req.email, req.password)
            return {"user": _user_payload(user), "access_token": token, "token_type": "bearer"}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(f"{settings.api_prefix}/auth/login")
    def auth_login(req: AuthRequest) -> dict:
        try:
            user, token = state.login(req.email, req.password)
            return {"user": _user_payload(user), "access_token": token, "token_type": "bearer"}
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    @app.get(f"{settings.api_prefix}/auth/me")
    def auth_me(authorization: str | None = Header(default=None)) -> dict:
        token = (authorization or "").removeprefix("Bearer ").strip()
        user = state.user_from_token(token)
        if not user:
            raise HTTPException(status_code=401, detail="Chưa đăng nhập hoặc token không hợp lệ")
        return {"user": _user_payload(user)}

    @app.post(f"{settings.api_prefix}/retrieve")
    def retrieve(req: RetrieveRequest) -> dict:
        from pathlib import Path

        hits = HybridRetriever(Path(req.index_dir)).retrieve(req.query, top_k=req.top_k)
        return {"hits": [hit.model_dump() for hit in hits]}

    @app.post(f"{settings.api_prefix}/context")
    def context(req: RetrieveRequest) -> dict:
        from pathlib import Path

        hits = HybridRetriever(Path(req.index_dir)).retrieve(req.query, top_k=req.top_k)
        return build_context(req.query, hits).model_dump()

    @app.post(f"{settings.api_prefix}/threads")
    def create_thread(user_id: str = "dev-user", title: str = "New chat") -> dict:
        thread = state.create_thread(user_id=user_id, title=title)
        return {"id": thread.id, "user_id": thread.user_id, "title": thread.title, "created_at": thread.created_at}

    @app.get(f"{settings.api_prefix}/threads")
    def list_threads(user_id: str = "dev-user") -> dict:
        return {"threads": [thread.__dict__ | {"messages": len(thread.messages)} for thread in state.list_threads(user_id)]}

    @app.get(f"{settings.api_prefix}/threads/{{thread_id}}/messages")
    def list_messages(thread_id: str) -> dict:
        thread = state.get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread không tồn tại")
        return {"messages": [msg.__dict__ for msg in thread.messages]}

    @app.post(f"{settings.api_prefix}/threads/{{thread_id}}/chat")
    def chat(thread_id: str, req: ChatRequest) -> dict:
        from pathlib import Path

        if not state.get_thread(thread_id):
            state.ensure_thread(thread_id, req.user_id, "Recovered chat")
        state.add_message(thread_id, "user", req.query)
        for key, value in extract_memory_facts(req.query).items():
            state.remember(req.user_id, key, value)
        hits = HybridRetriever(Path(req.index_dir)).retrieve(req.query, top_k=req.top_k) if _needs_retrieval(req.query) else []
        package = build_context(req.query, hits)
        answer = generate_answer(
            req.query,
            package,
            chat_history=state.recent_history(thread_id),
            user_memory=state.user_memory(req.user_id),
        )
        state.add_message(thread_id, "assistant", answer.answer, citations=answer.citations)
        return answer.model_dump()

    @app.post(f"{settings.api_prefix}/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        return state.add_feedback(req.user_id, req.message_id, req.rating, req.note)

    return app


app = create_app()


def _user_payload(user) -> dict:
    return {"id": user.id, "email": user.email, "created_at": user.created_at}


def _needs_retrieval(query: str) -> bool:
    q = query.lower().strip()
    legal_terms = (
        "phạt",
        "luật",
        "nghị định",
        "thông tư",
        "gplx",
        "blx",
        "giấy phép",
        "bằng lái",
        "sát hạch",
        "đủ tuổi",
        "thi",
        "xe máy",
        "ô tô",
        "mô tô",
        "đèn đỏ",
        "ngược chiều",
        "mũ bảo hiểm",
    )
    if any(term in q for term in legal_terms):
        return True
    personal_only_terms = ("tôi tên là", "tên tôi", "tôi là ai", "tôi tên là gì")
    if any(term in q for term in personal_only_terms):
        return False
    return len(q.split()) > 5

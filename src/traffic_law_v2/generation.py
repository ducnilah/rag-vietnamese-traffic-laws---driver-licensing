from __future__ import annotations

from uuid import uuid4
from urllib import request as urlrequest
import json

from langchain_core.prompts import ChatPromptTemplate

from traffic_law_v2.config import get_settings
from traffic_law_v2.models import AnswerPackage, ContextPackage


SYSTEM_PROMPT = """Bạn là trợ lý luật giao thông và giấy phép lái xe tại Việt Nam.
Trả lời tự nhiên, bằng tiếng Việt, như một trợ lý đang giúp người dùng thật.

Quy tắc bắt buộc:
- Chỉ dùng thông tin có trong Context cho phần nội dung pháp lý.
- Nếu Context đã có đáp án, phải trả lời thẳng vào câu hỏi; không từ chối, không nói chung chung, không khuyên liên hệ cơ quan chức năng.
- Ưu tiên nêu kết luận trước, sau đó nêu căn cứ pháp lý ngắn gọn và tự nhiên.
- Nếu Context chưa đủ dữ kiện để kết luận, nói rõ là chưa đủ căn cứ trong tài liệu hiện có.
- Không nhắc đến việc bạn là AI, không nói về giới hạn năng lực, không xin lỗi theo kiểu từ chối mẫu."""


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Thông tin nhớ về người dùng:\n{memory}\n\n"
            "Đoạn chat gần đây:\n{history}\n\n"
            "Câu hỏi hiện tại: {query}\n\n"
            "Context pháp lý:\n{context}\n\n"
            "Hãy trả lời thật ngắn gọn, đúng trọng tâm, dựa trên Context.\n"
            "Nếu Context có mức phạt, điều kiện, định nghĩa hoặc quy định liên quan, hãy nêu đáp án trực tiếp ngay câu đầu.\n"
            "Sau đó mới nêu căn cứ pháp lý ngắn gọn theo Điều/Chương/Mục nếu cần.\n"
            "Không được từ chối nếu Context đã đủ thông tin để trả lời.",
        ),
    ]
)


def generate_answer(
    query: str,
    context: ContextPackage,
    chat_history: str = "",
    user_memory: dict[str, str] | None = None,
) -> AnswerPackage:
    settings = get_settings()
    trace_id = uuid4().hex

    if settings.model_provider == "ollama":
        ollama_answer = _generate_with_ollama_native(
            settings=settings,
            query=query,
            context=context,
            chat_history=chat_history,
            user_memory=user_memory,
            strict=False,
        )
        if _looks_like_refusal(ollama_answer) and context.citations:
            ollama_answer = _generate_with_ollama_native(
                settings=settings,
                query=query,
                context=context,
                chat_history=chat_history,
                user_memory=user_memory,
                strict=True,
            )
        if ollama_answer:
            return AnswerPackage(
                answer=ollama_answer,
                citations=context.citations,
                confidence=context.confidence,
                model=settings.llm_model,
                trace_id=trace_id,
                fallback=False,
            )
        return _fallback_answer(context, settings.llm_model, trace_id)

    llm_conn = _resolve_llm_connection(settings)
    if llm_conn:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=llm_conn["api_key"],
            base_url=llm_conn["base_url"],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        try:
            memory_text = "\n".join(f"- {key}: {value}" for key, value in (user_memory or {}).items()) or "(chưa có)"
            history_text = chat_history.strip() or "(chưa có)"
            msg = (PROMPT | llm).invoke(
                {
                    "query": query,
                    "context": context.context_text,
                    "memory": memory_text,
                    "history": history_text,
                }
            )
            answer = str(msg.content).strip()
            return AnswerPackage(
                answer=answer,
                citations=context.citations,
                confidence=context.confidence,
                model=settings.llm_model,
                trace_id=trace_id,
                fallback=False,
            )
        except Exception:
            pass
    return _fallback_answer(context, settings.llm_model, trace_id)


def _resolve_llm_connection(settings) -> dict[str, str] | None:
    if settings.model_provider == "ollama":
        return None
    if settings.model_provider in {"openai", "openai_compatible"} and settings.openai_api_key:
        return {"api_key": settings.openai_api_key, "base_url": settings.openai_base_url or ""}
    return None


def _generate_with_ollama_native(
    settings,
    query: str,
    context: ContextPackage,
    chat_history: str,
    user_memory: dict[str, str] | None,
    strict: bool = False,
) -> str | None:
    try:
        memory_text = "\n".join(f"- {key}: {value}" for key, value in (user_memory or {}).items()) or "(chưa có)"
        history_text = chat_history.strip() or "(chưa có)"
        instruction = (
            "Hãy trả lời tự nhiên, tập trung đúng trọng tâm câu hỏi."
            " Nếu có căn cứ pháp lý thì nêu ngắn gọn ở cuối câu trả lời."
        )
        if strict:
            instruction = (
                "Context bên dưới da co du thong tin de tra loi.\n"
                "Hay tra loi truc tiep bang tieng Viet, dung thong tin trong Context.\n"
                "Khong duoc tu choi, khong noi chung chung, khong khuyen lien he co quan chuc nang.\n"
                "Neu la cau hoi ve muc phat, cau dau tien phai neu ro muc phat.\n"
                "Sau do neu can thi neu can cu phap ly ngan gon."
            )
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Thông tin nhớ về người dùng:\n{memory_text}\n\n"
            f"Đoạn chat gần đây:\n{history_text}\n\n"
            f"Câu hỏi hiện tại: {query}\n\n"
            f"Context pháp lý:\n{context.context_text}\n\n"
            f"{instruction}"
        )
        payload = {
            "model": settings.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_gpu": settings.ollama_num_gpu,
                "num_ctx": settings.ollama_num_ctx,
                "num_batch": settings.ollama_num_batch,
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        }
        base = settings.ollama_base_url
        native_url = base.replace("/v1", "") + "/api/generate" if base.endswith("/v1") else base.rstrip("/") + "/api/generate"
        req = urlrequest.Request(
            native_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        answer = str(body.get("response", "")).strip()
        return answer or None
    except Exception:
        return None


def _fallback_answer(
    context: ContextPackage,
    model: str,
    trace_id: str,
) -> AnswerPackage:
    if not context.hits:
        answer = "Mình chưa tìm thấy căn cứ phù hợp trong tài liệu để trả lời chắc chắn."
    else:
        lines = ["Hiện tại model chưa phản hồi nên mình chưa thể trả lời tự nhiên trong lượt này."]
        lines.append("Các căn cứ truy xuất được:")
        for _slot, citation in list(context.citations.items())[:5]:
            lines.append(f"- {_human_citation(citation)}")
        answer = "\n".join(lines)
    return AnswerPackage(
        answer=answer,
        citations=context.citations,
        confidence=context.confidence,
        model=f"{model}:fallback",
        trace_id=trace_id,
        fallback=True,
    )


def _looks_like_refusal(answer: str | None) -> bool:
    if not answer:
        return True
    lowered = answer.lower()
    markers = (
        "xin lỗi",
        "không thể cung cấp",
        "không thể trả lời",
        "không thể hỗ trợ",
        "liên hệ cơ quan chức năng",
        "tham khảo cơ quan chức năng",
        "tham khảo luật sư",
        "tôi không thể",
    )
    return any(marker in lowered for marker in markers)


def _human_citation(citation: dict) -> str:
    title = str(citation.get("document_title") or "Văn bản")
    article = str(citation.get("article") or "").strip()
    chapter = str(citation.get("chapter") or "").strip()
    section = str(citation.get("section") or "").strip()
    parts = [title]
    if article:
        parts.append(article)
    if chapter:
        parts.append(chapter)
    if section:
        parts.append(section)
    return " | ".join(parts)

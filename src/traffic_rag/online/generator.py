import json
import os
from dataclasses import dataclass
from typing import Dict, Optional
from urllib import request

from traffic_rag.online.context import ContextPackage


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model: str
    used_fallback: bool


def _fallback_answer(query: str, context: ContextPackage) -> str:
    lines = [
        "Dựa trên tài liệu truy xuất được, thông tin liên quan như sau:",
    ]
    for chunk in context.chunks[:3]:
        lines.append(f"- [{chunk.slot}] {chunk.citation}")
    lines.append(
        "Vui lòng đối chiếu các trích dẫn trên. Nếu cần, tôi có thể trích riêng đúng hàng/điều liên quan câu hỏi này."
    )
    return "\n".join(lines)


class ChatGenerator:
    def __init__(
        self,
        provider: str = "ollama",
        ollama_base_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "qwen2.5:1.5b-instruct",
        ollama_num_gpu: int = 0,
        ollama_num_ctx: int = 3072,
        ollama_num_batch: int = 32,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.ollama_num_gpu = ollama_num_gpu
        self.ollama_num_ctx = ollama_num_ctx
        self.ollama_num_batch = ollama_num_batch
        self.system_prompt = system_prompt or (
            "Bạn là Traffic Law & Driver Licensing Assistant cho Việt Nam.\n"
            "Nhiệm vụ: trả lời câu hỏi về luật giao thông đường bộ, xử phạt vi phạm, đào tạo/sát hạch/cấp GPLX.\n"
            "Ràng buộc:\n"
            "- Chỉ dùng thông tin trong Context được cung cấp.\n"
            "- Không suy diễn nếu Context không đủ; phải nói rõ chưa đủ căn cứ.\n"
            "- Khi nêu thông tin pháp lý phải gắn tham chiếu [C1], [C2]...\n"
            "- Luôn trả lời bằng tiếng Việt, không chèn tiếng Trung/Anh trừ khi người dùng yêu cầu.\n"
            "- Giọng điệu ngắn gọn, rõ ràng, đúng trọng tâm."
        )

    @classmethod
    def from_env(cls) -> "ChatGenerator":
        provider = os.getenv("TRAFFIC_RAG_LLM_PROVIDER", "ollama").strip().lower()
        base_url = os.getenv("TRAFFIC_RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
        model = os.getenv("TRAFFIC_RAG_OLLAMA_MODEL", "qwen2.5:1.5b-instruct").strip()
        num_gpu = int(os.getenv("TRAFFIC_RAG_OLLAMA_NUM_GPU", "0").strip())
        num_ctx = int(os.getenv("TRAFFIC_RAG_OLLAMA_NUM_CTX", "3072").strip())
        num_batch = int(os.getenv("TRAFFIC_RAG_OLLAMA_NUM_BATCH", "32").strip())
        system_prompt = os.getenv("TRAFFIC_RAG_SYSTEM_PROMPT", "").strip() or None
        return cls(
            provider=provider,
            ollama_base_url=base_url,
            ollama_model=model,
            ollama_num_gpu=num_gpu,
            ollama_num_ctx=num_ctx,
            ollama_num_batch=num_batch,
            system_prompt=system_prompt,
        )

    def generate(
        self,
        query: str,
        context: ContextPackage,
        conversation_context: str = "",
    ) -> GenerationResult:
        if self.provider == "ollama":
            out = self._generate_ollama(query, context, conversation_context=conversation_context)
            if out is not None:
                return out
        return GenerationResult(
            answer=_fallback_answer(query, context),
            model="fallback-template",
            used_fallback=True,
        )

    def _generate_ollama(
        self,
        query: str,
        context: ContextPackage,
        conversation_context: str = "",
    ) -> Optional[GenerationResult]:
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Conversation Memory (nếu có):\n{conversation_context.strip() or '(không có)'}\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context.context_text}\n\n"
            "Yêu cầu đầu ra:\n"
            "- Trả lời tự nhiên như trợ lý thật, xưng hô phù hợp.\n"
            "- Nếu người dùng hỏi thông tin cá nhân đã nói trước đó (ví dụ tên/tuổi), ưu tiên dùng Conversation Memory.\n"
            "- Với thông tin pháp lý/chuyên môn, chỉ dùng dữ kiện có trong Context và gắn tham chiếu [C#].\n"
            "- Nếu Context không đủ căn cứ, nói rõ chưa đủ căn cứ thay vì đoán.\n"
            "- Trả lời bằng tiếng Việt."
        )
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_gpu": self.ollama_num_gpu,
                "num_ctx": self.ollama_num_ctx,
                "num_batch": self.ollama_num_batch,
            },
        }
        req = request.Request(
            url=f"{self.ollama_base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                answer = str(data.get("response", "")).strip()
                if not answer:
                    return None
                return GenerationResult(
                    answer=answer,
                    model=self.ollama_model,
                    used_fallback=False,
                )
        except Exception:
            return None

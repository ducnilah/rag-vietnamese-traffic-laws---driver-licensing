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
        ollama_num_ctx: int = 1024,
        ollama_num_batch: int = 32,
    ) -> None:
        self.provider = provider
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.ollama_num_gpu = ollama_num_gpu
        self.ollama_num_ctx = ollama_num_ctx
        self.ollama_num_batch = ollama_num_batch

    @classmethod
    def from_env(cls) -> "ChatGenerator":
        provider = os.getenv("TRAFFIC_RAG_LLM_PROVIDER", "ollama").strip().lower()
        base_url = os.getenv("TRAFFIC_RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
        model = os.getenv("TRAFFIC_RAG_OLLAMA_MODEL", "qwen2.5:1.5b-instruct").strip()
        num_gpu = int(os.getenv("TRAFFIC_RAG_OLLAMA_NUM_GPU", "0").strip())
        num_ctx = int(os.getenv("TRAFFIC_RAG_OLLAMA_NUM_CTX", "1024").strip())
        num_batch = int(os.getenv("TRAFFIC_RAG_OLLAMA_NUM_BATCH", "32").strip())
        return cls(
            provider=provider,
            ollama_base_url=base_url,
            ollama_model=model,
            ollama_num_gpu=num_gpu,
            ollama_num_ctx=num_ctx,
            ollama_num_batch=num_batch,
        )

    def generate(self, query: str, context: ContextPackage) -> GenerationResult:
        if self.provider == "ollama":
            out = self._generate_ollama(query, context)
            if out is not None:
                return out
        return GenerationResult(
            answer=_fallback_answer(query, context),
            model="fallback-template",
            used_fallback=True,
        )

    def _generate_ollama(self, query: str, context: ContextPackage) -> Optional[GenerationResult]:
        prompt = (
            "Bạn là trợ lý luật giao thông Việt Nam. "
            "Chỉ trả lời dựa trên context được cung cấp. "
            "Nếu thiếu dữ liệu, nói rõ chưa đủ căn cứ. "
            "Luôn kèm tham chiếu [C1], [C2] tương ứng.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context.context_text}\n\n"
            "Trả lời ngắn gọn, chính xác."
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

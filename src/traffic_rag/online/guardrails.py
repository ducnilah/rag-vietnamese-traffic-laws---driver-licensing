import re
from dataclasses import dataclass
from typing import List


ALLOWED_DOMAIN_TERMS = {
    "giao thông",
    "lái xe",
    "giấy phép",
    "gplx",
    "hạng a1",
    "hạng a2",
    "hạng b1",
    "hạng b2",
    "sát hạch",
    "đào tạo lái xe",
    "mức phạt",
    "nồng độ cồn",
    "vượt đèn đỏ",
    "biển báo",
    "xe máy",
    "ô tô",
    "ngược chiều",
    "đường một chiều",
    "chở quá số người",
    "csgt",
    "luật giao thông",
    "xử phạt",
    "vi phạm",
    "điểm bằng",
    "tước bằng",
}

BLOCKED_POLICY_TERMS = {
    "làm giả",
    "mua bằng",
    "chạy chốt",
    "trốn phạt",
    "né phạt",
}

GREETING_TERMS = {
    "hi",
    "hello",
    "xin chào",
    "chào",
    "alo",
    "hey",
}


@dataclass(frozen=True)
class GuardrailResult:
    allow: bool
    code: str
    message: str
    risks: List[str]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def evaluate_query_guardrails(query: str) -> GuardrailResult:
    q = _normalize(query)
    risks: List[str] = []

    # Allow lightweight greetings/small talk so UX does not feel blocked.
    if q in GREETING_TERMS:
        return GuardrailResult(
            allow=True,
            code="OK_GREETING",
            message="ok",
            risks=risks,
        )

    if any(term in q for term in BLOCKED_POLICY_TERMS):
        risks.append("policy_violation")
        return GuardrailResult(
            allow=False,
            code="POLICY_BLOCKED",
            message="Yêu cầu liên quan hành vi vi phạm/chống đối pháp luật không được hỗ trợ.",
            risks=risks,
        )

    return GuardrailResult(
        allow=True,
        code="OK",
        message="ok",
        risks=risks,
    )

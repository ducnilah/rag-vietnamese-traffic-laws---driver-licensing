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
}

BLOCKED_POLICY_TERMS = {
    "làm giả",
    "mua bằng",
    "chạy chốt",
    "trốn phạt",
    "né phạt",
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

    if any(term in q for term in BLOCKED_POLICY_TERMS):
        risks.append("policy_violation")
        return GuardrailResult(
            allow=False,
            code="POLICY_BLOCKED",
            message="Yêu cầu liên quan hành vi vi phạm/chống đối pháp luật không được hỗ trợ.",
            risks=risks,
        )

    if not any(term in q for term in ALLOWED_DOMAIN_TERMS):
        risks.append("out_of_domain")
        return GuardrailResult(
            allow=False,
            code="OUT_OF_DOMAIN",
            message="Câu hỏi ngoài phạm vi trợ lý luật giao thông và sát hạch lái xe.",
            risks=risks,
        )

    return GuardrailResult(
        allow=True,
        code="OK",
        message="ok",
        risks=risks,
    )

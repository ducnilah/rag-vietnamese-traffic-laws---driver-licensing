from __future__ import annotations

import re
from typing import Iterable

TOKEN_RE = re.compile(r"[\wÀ-ỹĐđ]+", re.UNICODE)
STOPWORDS = {
    "là", "và", "của", "có", "cho", "các", "một", "những", "người", "điều", "khiển",
    "trong", "theo", "đối", "với", "thì", "bị", "bao", "nhiêu", "như", "thế", "nào",
    "khi", "được", "không", "phải", "về", "tại", "này", "đó", "hoặc", "nếu",
}


def tokens(text: str, *, remove_stopwords: bool = True) -> list[str]:
    raw = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    if remove_stopwords:
        return [tok for tok in raw if len(tok) >= 2 and tok not in STOPWORDS]
    return raw


def token_prf(candidate: str, reference: str) -> dict[str, float]:
    cand = tokens(candidate)
    ref = tokens(reference)
    if not cand or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    cand_counts = _counts(cand)
    ref_counts = _counts(ref)
    overlap = sum(min(cand_counts.get(tok, 0), ref_counts.get(tok, 0)) for tok in cand_counts)
    precision = overlap / max(1, len(cand))
    recall = overlap / max(1, len(ref))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def term_coverage(text: str, terms: Iterable[str]) -> dict[str, object]:
    normalized = text.lower()
    expected = [term for term in terms if term]
    matched = [term for term in expected if term.lower() in normalized]
    return {
        "matched": matched,
        "missing": [term for term in expected if term not in matched],
        "score": round(len(matched) / max(1, len(expected)), 4),
    }


def _counts(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out

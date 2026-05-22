from __future__ import annotations

import re
import unicodedata
from hashlib import sha1
from typing import Iterable, List


TOKEN_RE = re.compile(r"[\wÀ-ỹĐđ]+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def stable_id(*parts: str, length: int = 16) -> str:
    raw = "::".join(parts).encode("utf-8")
    return sha1(raw).hexdigest()[:length]


def chunk_by_tokens(text: str, target_tokens: int = 700, overlap: int = 100) -> Iterable[str]:
    words = text.split()
    if not words:
        return
    step = max(1, target_tokens - overlap)
    for start in range(0, len(words), step):
        end = min(len(words), start + target_tokens)
        yield " ".join(words[start:end]).strip()
        if end >= len(words):
            break


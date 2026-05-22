from __future__ import annotations

import hashlib
import math
from typing import Iterable, List

from traffic_law_v2.config import Settings
from traffic_law_v2.text import tokenize


class HashEmbedding:
    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed_query(self, text: str) -> List[float]:
        vec = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]


def get_embeddings(settings: Settings):
    if settings.model_provider == "ollama":
        # In v2 we keep local deterministic embeddings for Ollama mode so users
        # can run fully local generation without requiring an external embedding API.
        return HashEmbedding()
    if settings.model_provider in {"openai", "openai_compatible"} and settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return HashEmbedding()


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))

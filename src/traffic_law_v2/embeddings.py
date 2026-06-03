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


class SentenceTransformerEmbedding:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 4) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for local embeddings. "
                "Install with: python -m pip install sentence-transformers torch"
            ) from exc

        kwargs = {"device": device} if device else {}
        self.model = SentenceTransformer(model_name, **kwargs)
        self.batch_size = batch_size

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        rows = list(texts)
        if not rows:
            return []
        vectors = self.model.encode(
            rows,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


def get_embeddings(settings: Settings):
    if settings.embedding_provider == "hash":
        return HashEmbedding()
    if settings.embedding_provider == "local":
        return SentenceTransformerEmbedding(
            settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
    if settings.embedding_provider in {"openai", "openai_compatible"} and settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        base_url = settings.openai_base_url if settings.embedding_provider == "openai_compatible" else None
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=base_url,
        )
    return HashEmbedding()


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))

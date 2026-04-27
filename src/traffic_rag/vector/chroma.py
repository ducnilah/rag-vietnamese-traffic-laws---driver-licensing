import logging
import math
import os
import zlib
from pathlib import Path
from typing import Dict, List, Optional

from traffic_rag.offline.bm25 import tokenize
from traffic_rag.offline.models import Chunk

try:
    import chromadb

    CHROMADB_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on local environment
    chromadb = None  # type: ignore[assignment]
    CHROMADB_IMPORT_ERROR = exc

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on local environment
    SentenceTransformer = None  # type: ignore[assignment]
    SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

CHROMA_DIRNAME_DEFAULT = "chroma"
CHROMA_COLLECTION_DEFAULT = "traffic_law_chunks"
CHROMA_EMBEDDING_BACKEND_DEFAULT = "bge-m3"
CHROMA_EMBEDDING_MODEL_DEFAULT = "BAAI/bge-m3"


def chromadb_available() -> bool:
    return chromadb is not None


def _normalize_vector(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


class HashEmbeddingFunction:
    """
    Lightweight deterministic embedding for local ChromaDB usage.
    This keeps the project runnable without downloading heavyweight models.
    """

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.dim = dim

    def name(self) -> str:
        # Chroma custom embedding function contract (newer versions) expects name().
        return f"hash-embedding-{self.dim}"

    def get_config(self) -> Dict[str, int]:
        # Some Chroma versions serialize embedding functions via config.
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: Dict[str, int]) -> "HashEmbeddingFunction":
        return HashEmbeddingFunction(dim=int(config.get("dim", 384)))

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            idx = zlib.crc32(token.encode("utf-8")) % self.dim
            vec[idx] += 1.0
        return _normalize_vector(vec)

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return [self._embed_one(text) for text in input]

    def embed_documents(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return self.__call__(input)

    def embed_query(self, input: str) -> List[float]:  # noqa: A002
        return self._embed_one(input)


class BgeM3EmbeddingFunction:
    def __init__(
        self,
        model_name: str = CHROMA_EMBEDDING_MODEL_DEFAULT,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 8,
    ) -> None:
        if SentenceTransformer is None:
            error = (
                "sentence-transformers is not installed. Install with: "
                "python3 -m pip install sentence-transformers torch"
            )
            if SENTENCE_TRANSFORMERS_IMPORT_ERROR:
                error += f" (import error: {SENTENCE_TRANSFORMERS_IMPORT_ERROR})"
            raise RuntimeError(error)
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = max(1, int(batch_size))
        self.model = (
            SentenceTransformer(model_name, device=device)
            if device
            else SentenceTransformer(model_name)
        )

    def name(self) -> str:
        return f"bge-m3-{self.model_name}"

    def get_config(self) -> Dict[str, object]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
        }

    @staticmethod
    def build_from_config(config: Dict[str, object]) -> "BgeM3EmbeddingFunction":
        return BgeM3EmbeddingFunction(
            model_name=str(config.get("model_name", CHROMA_EMBEDDING_MODEL_DEFAULT)),
            device=(
                None
                if config.get("device") in (None, "None", "")
                else str(config.get("device"))
            ),
            normalize_embeddings=bool(config.get("normalize_embeddings", True)),
            batch_size=int(config.get("batch_size", 8)),
        )

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        vectors = self.model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            batch_size=self.batch_size,
        )
        return vectors.tolist()

    def embed_documents(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        return self.__call__(input)

    def embed_query(self, input: str) -> List[float]:  # noqa: A002
        return self.__call__([input])[0]


def build_embedding_function(
    backend: str = CHROMA_EMBEDDING_BACKEND_DEFAULT,
    model_name: str = CHROMA_EMBEDDING_MODEL_DEFAULT,
    hash_dim: int = 384,
) -> object:
    if backend == "bge-m3":
        device = os.getenv("TRAFFIC_RAG_EMBED_DEVICE")
        if device == "":
            device = None
        batch_size_raw = os.getenv("TRAFFIC_RAG_EMBED_BATCH_SIZE", "8")
        try:
            batch_size = max(1, int(batch_size_raw))
        except ValueError:
            batch_size = 8
        return BgeM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
    if backend == "hash":
        return HashEmbeddingFunction(dim=hash_dim)
    raise ValueError(f"Unsupported embedding backend: {backend}")


def _sanitize_metadata(chunk: Chunk) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "doc_id": chunk.doc_id,
        "start_char": int(chunk.start_char),
        "end_char": int(chunk.end_char),
    }
    for key, value in chunk.metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[str(key)] = value
        else:
            metadata[str(key)] = str(value)
    return metadata


class ChromaIndexer:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = CHROMA_COLLECTION_DEFAULT,
        embedding_backend: str = CHROMA_EMBEDDING_BACKEND_DEFAULT,
        embedding_model: str = CHROMA_EMBEDDING_MODEL_DEFAULT,
        embedding_dim: int = 384,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_backend = embedding_backend
        self.embedding_model = embedding_model
        self.embedding_fn = build_embedding_function(
            backend=embedding_backend,
            model_name=embedding_model,
            hash_dim=embedding_dim,
        )

    def build(self, chunks: List[Chunk], reset_collection: bool = True) -> Dict[str, object]:
        if chromadb is None:
            error = (
                "chromadb is not installed. Install with: "
                "python3 -m pip install chromadb"
            )
            if CHROMADB_IMPORT_ERROR:
                error += f" (import error: {CHROMADB_IMPORT_ERROR})"
            raise RuntimeError(error)

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_dir))

        if reset_collection:
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass

        collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={
                "hnsw:space": "cosine",
                "embedding_backend": self.embedding_backend,
                "embedding_model": self.embedding_model,
            },
        )

        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.upsert(
                ids=[row.chunk_id for row in batch],
                documents=[row.text for row in batch],
                metadatas=[_sanitize_metadata(row) for row in batch],
            )

        logger.info(
            "chroma_index_built collection=%s vectors=%d path=%s",
            self.collection_name,
            len(chunks),
            self.persist_dir,
        )
        return {
            "enabled": True,
            "collection": self.collection_name,
            "vectors": len(chunks),
            "persist_dir": str(self.persist_dir),
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
        }

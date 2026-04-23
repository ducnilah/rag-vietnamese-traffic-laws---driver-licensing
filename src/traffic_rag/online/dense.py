import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Set

from traffic_rag.offline.bm25 import tokenize
from traffic_rag.online.store import ChunkStore
from traffic_rag.vector.chroma import CHROMA_COLLECTION_DEFAULT, HashEmbeddingFunction, chromadb

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DenseHit:
    chunk_id: str
    score: float


class DenseRetriever(Protocol):
    def search(self, query: str, top_k: int = 30) -> List[DenseHit]:
        ...


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    if inter == 0:
        return 0.0
    union = len(a.union(b))
    return inter / float(union)


class JaccardDenseRetriever:
    """
    Dense skeleton for M2:
    lexical proxy so hybrid architecture can be tested before real embeddings.
    """

    def __init__(self, store: ChunkStore) -> None:
        self.store = store
        self.chunk_tokens: Dict[str, Set[str]] = {
            row.chunk_id: set(tokenize(row.text)) for row in store.all()
        }

    def search(self, query: str, top_k: int = 30) -> List[DenseHit]:
        q_tokens = set(tokenize(query))
        scored: List[DenseHit] = []
        for chunk_id, c_tokens in self.chunk_tokens.items():
            score = _jaccard(q_tokens, c_tokens)
            if score > 0:
                scored.append(DenseHit(chunk_id=chunk_id, score=score))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


class ChromaDenseRetriever:
    def __init__(
        self,
        chroma_dir: Path,
        collection_name: str = CHROMA_COLLECTION_DEFAULT,
        embedding_dim: int = 384,
    ) -> None:
        if chromadb is None:
            raise RuntimeError(
                "chromadb is not installed. Install with: python3 -m pip install chromadb"
            )
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.embedding_fn = HashEmbeddingFunction(dim=embedding_dim)
        # Use precomputed query embeddings at search time to avoid runtime
        # incompatibilities across Chroma embedding-function interfaces.
        self.collection = self.client.get_collection(name=collection_name)
        self.collection_name = collection_name

    @classmethod
    def try_create(
        cls,
        chroma_dir: Path,
        collection_name: str = CHROMA_COLLECTION_DEFAULT,
        embedding_dim: int = 384,
    ) -> Optional["ChromaDenseRetriever"]:
        try:
            return cls(
                chroma_dir=chroma_dir,
                collection_name=collection_name,
                embedding_dim=embedding_dim,
            )
        except Exception as exc:
            logger.warning(
                "chroma_dense_unavailable dir=%s collection=%s reason=%s",
                chroma_dir,
                collection_name,
                exc,
            )
            return None

    def search(self, query: str, top_k: int = 30) -> List[DenseHit]:
        query_vec = self.embedding_fn.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["distances"],
        )
        ids: List[str] = []
        distances: List[float] = []
        if results.get("ids"):
            ids = [str(item) for item in results["ids"][0]]
        if results.get("distances"):
            distances = [float(item) for item in results["distances"][0]]

        hits: List[DenseHit] = []
        for chunk_id, distance in zip(ids, distances):
            score = max(0.0, 1.0 - distance)
            hits.append(DenseHit(chunk_id=chunk_id, score=score))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

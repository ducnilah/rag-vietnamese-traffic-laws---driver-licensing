from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from traffic_law_v2.config import get_settings
from traffic_law_v2.embeddings import get_embeddings
from traffic_law_v2.models import Chunk


COLLECTION_NAME = "traffic_law_v2_chunks"


def build_chroma_index(chunks: List[Chunk], index_dir: Path, collection_name: str = COLLECTION_NAME) -> dict:
    """Persist dense vectors to Chroma.

    We still keep JSONL artifacts as the source of truth. Chroma is the dense
    ANN layer used by the serving path when the project grows beyond local
    smoke-test scale.
    """
    import chromadb

    persist_dir = index_dir / "chroma"
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
    embeddings = get_embeddings(get_settings())
    vectors = embeddings.embed_documents([chunk.text for chunk in chunks]) if chunks else []
    if chunks:
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[_metadata(chunk) for chunk in chunks],
        )
    return {
        "enabled": True,
        "collection": collection_name,
        "vectors": len(chunks),
        "persist_dir": str(persist_dir),
    }


def search_chroma(index_dir: Path, query: str, top_k: int, collection_name: str = COLLECTION_NAME) -> List[tuple[str, float]]:
    import chromadb

    persist_dir = index_dir / "chroma"
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)
    embeddings = get_embeddings(get_settings())
    qv = embeddings.embed_query(query)
    result = collection.query(query_embeddings=[qv], n_results=top_k, include=["distances"])
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    # Chroma cosine distance is lower-is-better. Convert to a similarity-ish score.
    return [(cid, 1.0 - float(distance)) for cid, distance in zip(ids, distances)]


def _metadata(chunk: Chunk) -> Dict[str, str | int | float | bool]:
    meta = chunk.metadata.model_dump()
    clean: Dict[str, str | int | float | bool] = {
        "doc_id": chunk.doc_id,
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
    }
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean

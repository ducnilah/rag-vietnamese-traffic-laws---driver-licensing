from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from traffic_law_v2.ingestion import chunk_documents, load_documents
from traffic_law_v2.models import Chunk
from traffic_law_v2.text import tokenize
from traffic_law_v2.vectorstore import build_chroma_index


class SimpleBM25:
    def __init__(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        self.doc_tokens = [tokenize(c.text) for c in chunks]
        self.avgdl = sum(len(t) for t in self.doc_tokens) / max(1, len(self.doc_tokens))
        self.df: Dict[str, int] = defaultdict(int)
        for toks in self.doc_tokens:
            for token in set(toks):
                self.df[token] += 1

    def search(self, query: str, top_k: int = 10) -> List[tuple[str, float]]:
        q = tokenize(query)
        scores: List[tuple[str, float]] = []
        n = len(self.doc_tokens)
        for chunk, toks in zip(self.chunks, self.doc_tokens):
            counts = Counter(toks)
            dl = len(toks)
            score = 0.0
            for term in q:
                if term not in counts:
                    continue
                df = self.df.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = counts[term]
                score += idf * (tf * 2.2) / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / max(1.0, self.avgdl)))
            if score > 0:
                scores.append((chunk.chunk_id, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]


def build_index(raw_dir: Path, index_dir: Path) -> dict:
    docs = load_documents(raw_dir)
    chunks = chunk_documents(docs)
    index_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(index_dir / "documents.jsonl", [d.model_dump() for d in docs])
    _write_jsonl(index_dir / "chunks.jsonl", [c.model_dump() for c in chunks])
    _write_jsonl(
        index_dir / "chunks_legal.jsonl",
        [
            {
                "chapter": c.metadata.chapter,
                "article": c.metadata.article,
                "clause": c.metadata.clause,
                "point": c.metadata.point,
                "content": c.text,
            }
            for c in chunks
        ],
    )
    bm25 = SimpleBM25(chunks)
    (index_dir / "bm25.json").write_text(
        json.dumps(
            {
                "chunk_ids": [c.chunk_id for c in chunks],
                "avgdl": bm25.avgdl,
                "df": dict(bm25.df),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report = {
        "documents": len(docs),
        "chunks": len(chunks),
        "warnings": _quality_warnings(chunks),
    }
    try:
        report["chroma"] = build_chroma_index(chunks, index_dir)
    except Exception as exc:
        # Chroma is important for production, but JSONL+BM25 remains enough for
        # local debugging. We surface the issue without losing the index build.
        report["chroma"] = {"enabled": False, "error": str(exc)}
    (index_dir / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_chunks(index_dir: Path) -> List[Chunk]:
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        return []
    return [Chunk.model_validate(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _quality_warnings(chunks: List[Chunk]) -> List[dict]:
    warnings: List[dict] = []
    seen = set()
    for chunk in chunks:
        if chunk.token_count < 30:
            warnings.append({"code": "SHORT_CHUNK", "chunk_id": chunk.chunk_id})
        fingerprint = " ".join(tokenize(chunk.text)[:80])
        if fingerprint in seen:
            warnings.append({"code": "POSSIBLE_DUPLICATE", "chunk_id": chunk.chunk_id})
        seen.add(fingerprint)
    return warnings

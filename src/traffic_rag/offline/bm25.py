import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .models import Chunk


def tokenize(text: str) -> List[str]:
    return re.findall(r"[\w\-]+", text.lower())


@dataclass(frozen=True)
class BM25Hit:
    chunk_id: str
    score: float


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.doc_term_freqs: Dict[str, Dict[str, int]] = {}
        self.df: Dict[str, int] = defaultdict(int)
        self.idf: Dict[str, float] = {}

    def build(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build BM25 on empty chunk list")

        total_len = 0
        self.doc_lengths = {}
        self.doc_term_freqs = {}
        self.df = defaultdict(int)

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tf = Counter(tokens)
            self.doc_term_freqs[chunk.chunk_id] = dict(tf)
            doc_len = len(tokens)
            self.doc_lengths[chunk.chunk_id] = doc_len
            total_len += doc_len
            for term in tf:
                self.df[term] += 1

        n_docs = len(chunks)
        self.avg_doc_len = total_len / n_docs if n_docs else 0.0
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(((n_docs - freq + 0.5) / (freq + 0.5)) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[BM25Hit]:
        if not self.doc_term_freqs:
            raise RuntimeError("BM25 index is not built")

        q_terms = tokenize(query)
        scores: Dict[str, float] = defaultdict(float)

        for term in q_terms:
            term_idf = self.idf.get(term)
            if term_idf is None:
                continue
            for doc_id, tf_map in self.doc_term_freqs.items():
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                dl = self.doc_lengths[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / self.avg_doc_len))
                scores[doc_id] += term_idf * (tf * (self.k1 + 1)) / denom

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [BM25Hit(chunk_id=doc_id, score=score) for doc_id, score in ranked[:top_k]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_lengths": self.doc_lengths,
            "avg_doc_len": self.avg_doc_len,
            "doc_term_freqs": self.doc_term_freqs,
            "df": dict(self.df),
            "idf": self.idf,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "BM25Index":
        obj = cls(k1=float(data["k1"]), b=float(data["b"]))
        obj.doc_lengths = {str(k): int(v) for k, v in dict(data["doc_lengths"]).items()}
        obj.avg_doc_len = float(data["avg_doc_len"])
        obj.doc_term_freqs = {
            str(k): {str(t): int(c) for t, c in dict(v).items()}
            for k, v in dict(data["doc_term_freqs"]).items()
        }
        obj.df = defaultdict(int, {str(k): int(v) for k, v in dict(data["df"]).items()})
        obj.idf = {str(k): float(v) for k, v in dict(data["idf"]).items()}
        return obj

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

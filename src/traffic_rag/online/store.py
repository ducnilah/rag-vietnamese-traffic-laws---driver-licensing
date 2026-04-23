import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: Dict[str, str]


class ChunkStore:
    """
    In-memory access layer for chunks.jsonl.
    """

    def __init__(self, rows: List[ChunkRecord]) -> None:
        self._rows = rows
        self._by_id = {row.chunk_id: row for row in rows}

    @classmethod
    def load(cls, chunks_path: Path) -> "ChunkStore":
        rows: List[ChunkRecord] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                rows.append(
                    ChunkRecord(
                        chunk_id=str(raw["chunk_id"]),
                        doc_id=str(raw["doc_id"]),
                        text=str(raw.get("text", "")),
                        start_char=int(raw.get("start_char", 0)),
                        end_char=int(raw.get("end_char", 0)),
                        metadata=dict(raw.get("metadata", {})),
                    )
                )
        return cls(rows)

    def get(self, chunk_id: str) -> ChunkRecord:
        return self._by_id[chunk_id]

    def all(self) -> List[ChunkRecord]:
        return self._rows

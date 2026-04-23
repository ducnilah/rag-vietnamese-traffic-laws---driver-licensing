from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    title: str
    text: str
    source_path: str
    version: str = "v1"
    created_at: str = field(default_factory=utc_now_iso)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: Dict[str, str]

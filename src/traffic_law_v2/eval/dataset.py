from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EvalItem:
    id: str
    query: str
    reference_answer: str
    category: str = "general"
    expected_terms: tuple[str, ...] = field(default_factory=tuple)
    expected_citations: tuple[str, ...] = field(default_factory=tuple)


def load_eval_dataset(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw: dict[str, Any] = json.loads(line)
            items.append(
                EvalItem(
                    id=str(raw.get("id") or f"case_{line_no}"),
                    category=str(raw.get("category") or "general"),
                    query=str(raw["query"]),
                    reference_answer=str(raw["reference_answer"]),
                    expected_terms=tuple(str(x) for x in raw.get("expected_terms", [])),
                    expected_citations=tuple(str(x) for x in raw.get("expected_citations", [])),
                )
            )
    return items


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

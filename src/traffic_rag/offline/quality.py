import re
from dataclasses import dataclass
from typing import Dict, List

from .models import Chunk


@dataclass(frozen=True)
class QualityIssue:
    level: str
    chunk_id: str
    code: str
    message: str


@dataclass(frozen=True)
class QualityReport:
    total_chunks: int
    warnings: int
    errors: int
    issues: List[QualityIssue]

    @property
    def passed(self) -> bool:
        return self.errors == 0


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def run_quality_checks(
    chunks: List[Chunk],
    min_chars: int = 80,
    max_chars: int = 1800,
    min_chars_table: int = 40,
    max_chars_table: int = 5000,
) -> QualityReport:
    issues: List[QualityIssue] = []
    seen_text: Dict[str, str] = {}

    for chunk in chunks:
        is_table = chunk.metadata.get("content_type") == "table"
        effective_min = min_chars_table if is_table else min_chars
        effective_max = max_chars_table if is_table else max_chars
        length = len(chunk.text)
        if length < effective_min:
            issues.append(
                QualityIssue(
                    level="warning",
                    chunk_id=chunk.chunk_id,
                    code="short_chunk",
                    message=f"Chunk shorter than {effective_min} chars ({length}).",
                )
            )
        if length > effective_max:
            issues.append(
                QualityIssue(
                    level="error",
                    chunk_id=chunk.chunk_id,
                    code="oversized_chunk",
                    message=f"Chunk longer than {effective_max} chars ({length}).",
                )
            )

        normalized = _normalized(chunk.text)
        if normalized in seen_text:
            issues.append(
                QualityIssue(
                    level="warning",
                    chunk_id=chunk.chunk_id,
                    code="duplicate_chunk",
                    message=f"Possible duplicate of {seen_text[normalized]}.",
                )
            )
        else:
            seen_text[normalized] = chunk.chunk_id

    warnings = sum(1 for item in issues if item.level == "warning")
    errors = sum(1 for item in issues if item.level == "error")
    return QualityReport(
        total_chunks=len(chunks),
        warnings=warnings,
        errors=errors,
        issues=issues,
    )

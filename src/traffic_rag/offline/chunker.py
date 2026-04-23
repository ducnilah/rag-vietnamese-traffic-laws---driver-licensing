import hashlib
import re
from typing import Dict, List, Optional, Tuple

from .models import Chunk, SourceDocument

LEGAL_HEADING_RE = re.compile(
    r"(?im)^(?:\s*\[SECTION\s+\d+\].*|\s*Phần\s+[IVXLCDM\d]+.*|\s*Chương\s+[IVXLCDM\d]+.*|\s*Mục\s+\d+.*|\s*Điều\s+\d+[A-Za-z]?[.:].*)$"
)
ARTICLE_RE = re.compile(r"(?im)^\s*Điều\s+(?P<no>\d+[A-Za-z]?)\b")
CHAPTER_RE = re.compile(r"(?im)^\s*Chương\s+(?P<no>[IVXLCDM\d]+)\b")
PART_RE = re.compile(r"(?im)^\s*Phần\s+(?P<no>[IVXLCDM\d]+)\b")
INSTRUMENT_LINE_RE = re.compile(
    r"(?im)\b(?P<kind>Luật|Nghị định|Thông tư)\s*(?:số\s*)?(?P<no>\d+/\d{4}/[A-ZĐ\-]+)?"
)
INSTRUMENT_NO_RE = re.compile(r"(?im)\b(?P<no>\d+/\d{4}/[A-ZĐ\-]+)\b")


def _paragraph_spans(text: str) -> List[Tuple[int, int]]:
    spans = []
    for match in re.finditer(r"\S[\s\S]*?(?=\n\n|$)", text):
        spans.append((match.start(), match.end()))
    return spans


def _section_spans(text: str) -> List[Tuple[int, int]]:
    matches = list(LEGAL_HEADING_RE.finditer(text))
    if not matches:
        return []

    starts = [m.start() for m in matches]
    spans: List[Tuple[int, int]] = []

    if starts[0] > 0 and text[: starts[0]].strip():
        spans.append((0, starts[0]))

    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        if text[start:end].strip():
            spans.append((start, end))
    return spans


def _group_spans(spans: List[Tuple[int, int]], target_chars: int) -> List[Tuple[int, int]]:
    if not spans:
        return []
    grouped: List[Tuple[int, int]] = []
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if (end - cur_start) <= target_chars:
            cur_end = end
        else:
            grouped.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    grouped.append((cur_start, cur_end))
    return grouped


def _split_oversized_span(
    text: str,
    start: int,
    end: int,
    target_chars: int,
) -> List[Tuple[int, int]]:
    if (end - start) <= target_chars:
        return [(start, end)]

    spans: List[Tuple[int, int]] = []
    cursor = start
    while cursor < end:
        cut = min(cursor + target_chars, end)
        if cut < end:
            # Prefer natural boundaries so chunks are easier to read/retrieve.
            boundary = text.rfind("\n", cursor, cut)
            if boundary > cursor + (target_chars // 2):
                cut = boundary
        spans.append((cursor, cut))
        if cut == cursor:
            break
        cursor = cut
    return spans


def _extract_legal_metadata(text: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return metadata

    first = lines[0]
    if len(first) <= 200:
        metadata["section_title"] = first
    else:
        metadata["section_title"] = first[:200]

    m_article = ARTICLE_RE.search(text)
    if m_article:
        metadata["article_no"] = m_article.group("no")

    m_chapter = CHAPTER_RE.search(text)
    if m_chapter:
        metadata["chapter_no"] = m_chapter.group("no")

    m_part = PART_RE.search(text)
    if m_part:
        metadata["part_no"] = m_part.group("no")

    return metadata


def _last_heading_no_before(text: str, pattern: re.Pattern, end_pos: int) -> Optional[str]:
    last: Optional[str] = None
    for match in pattern.finditer(text[:end_pos]):
        last = match.group("no")
    return last


def _infer_document_instrument(text: str, fallback_title: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    text_head = text[:12000]
    n_match = INSTRUMENT_NO_RE.search(text_head)
    number = n_match.group("no").strip() if n_match else ""

    kind = ""
    kind_matches = list(INSTRUMENT_LINE_RE.finditer(text_head))
    if kind_matches and n_match:
        no_pos = n_match.start()
        # Prefer kind appearing right after document number (e.g. "Số ... THÔNG TƯ").
        after = [m for m in kind_matches if no_pos <= m.start() <= no_pos + 500]
        if after:
            kind = (after[0].group("kind") or "").strip()
        else:
            nearest = min(kind_matches, key=lambda m: abs(m.start() - no_pos))
            kind = (nearest.group("kind") or "").strip()
    elif kind_matches:
        kind = (kind_matches[0].group("kind") or "").strip()

    if kind or number:
        if kind:
            metadata["instrument_type"] = kind
        if number:
            metadata["instrument_no"] = number
        if kind and number:
            metadata["instrument_ref"] = f"{kind} {number}"
        elif kind:
            metadata["instrument_ref"] = kind
        return metadata

    # Fallback to title/file stem when legal line is not found.
    if fallback_title:
        metadata["instrument_ref"] = fallback_title
    return metadata


def semantic_chunk_document(
    doc: SourceDocument,
    target_chars: int = 700,
    overlap_chars: int = 120,
    base_metadata: Optional[Dict[str, str]] = None,
) -> List[Chunk]:
    if not doc.text.strip():
        return []

    spans = _section_spans(doc.text)
    if not spans:
        spans = _paragraph_spans(doc.text)
    if not spans:
        return []

    doc_instrument_meta = _infer_document_instrument(doc.text, doc.title)

    grouped_with_floor: List[Tuple[int, int, int]] = []
    for coarse_start, coarse_end in spans:
        normalized = _split_oversized_span(doc.text, coarse_start, coarse_end, target_chars)
        grouped = _group_spans(normalized, target_chars)
        for start, end in grouped:
            grouped_with_floor.append((start, end, coarse_start))

    chunks = []
    for idx, (start, end, coarse_start) in enumerate(grouped_with_floor):
        final_start = max(coarse_start, start - overlap_chars)
        text = doc.text[final_start:end].strip()
        digest = hashlib.sha1(f"{doc.doc_id}:{idx}:{final_start}:{end}".encode("utf-8")).hexdigest()[:12]
        chunk_id = f"{doc.doc_id}-c{idx}-{digest}"
        metadata = {
            "title": doc.title,
            "source_path": doc.source_path,
            "version": doc.version,
            "chunk_index": str(idx),
        }
        legal_meta = _extract_legal_metadata(text)
        if "chapter_no" not in legal_meta:
            prev_chapter = _last_heading_no_before(doc.text, CHAPTER_RE, start)
            if prev_chapter:
                legal_meta["chapter_no"] = prev_chapter
        if "part_no" not in legal_meta:
            prev_part = _last_heading_no_before(doc.text, PART_RE, start)
            if prev_part:
                legal_meta["part_no"] = prev_part

        metadata.update(doc_instrument_meta)
        metadata.update(legal_meta)
        if base_metadata:
            metadata.update(base_metadata)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                text=text,
                start_char=final_start,
                end_char=end,
                metadata=metadata,
            )
        )
    return chunks


def semantic_chunk_documents(
    docs: List[SourceDocument],
    target_chars: int = 700,
    overlap_chars: int = 120,
    base_metadata: Optional[Dict[str, str]] = None,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in docs:
        chunks.extend(
            semantic_chunk_document(
                doc,
                target_chars,
                overlap_chars,
                base_metadata=base_metadata,
            )
        )
    return chunks

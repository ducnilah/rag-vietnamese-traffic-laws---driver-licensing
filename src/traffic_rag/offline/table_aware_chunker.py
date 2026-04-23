import hashlib
import re
from dataclasses import replace
from typing import Dict, List

from .chunker import semantic_chunk_document
from .models import Chunk, SourceDocument

TABLE_BLOCK_RE = re.compile(
    r"(?m)^\[TABLE (?P<table_id>\d+\.\d+)\][^\n]*\n(?P<table>(?:\|[^\n]*\n)+)"
)
INSTRUMENT_LINE_RE = re.compile(
    r"(?im)\b(?P<kind>Luật|Nghị định|Thông tư)\s*(?:số\s*)?(?P<no>\d+/\d{4}/[A-ZĐ\-]+)?"
)
INSTRUMENT_NO_RE = re.compile(r"(?im)\b(?P<no>\d+/\d{4}/[A-ZĐ\-]+)\b")


def _infer_document_instrument(text: str, fallback_title: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    text_head = text[:12000]
    n_match = INSTRUMENT_NO_RE.search(text_head)
    number = n_match.group("no").strip() if n_match else ""

    kind = ""
    kind_matches = list(INSTRUMENT_LINE_RE.finditer(text_head))
    if kind_matches and n_match:
        no_pos = n_match.start()
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

    if fallback_title:
        metadata["instrument_ref"] = fallback_title
    return metadata


def _make_table_chunk(
    doc: SourceDocument,
    table_id: str,
    table_body: str,
    chunk_index: int,
    start_char: int,
    end_char: int,
    doc_instrument_meta: Dict[str, str],
) -> Chunk:
    digest = hashlib.sha1(f"{doc.doc_id}:table:{table_id}:{start_char}:{end_char}".encode("utf-8")).hexdigest()[:12]
    page = table_id.split(".", 1)[0]
    table_text = f"[TABLE {table_id}]\n{table_body.strip()}"
    metadata = {
        "title": doc.title,
        "source_path": doc.source_path,
        "version": doc.version,
        "chunk_index": str(chunk_index),
        "content_type": "table",
        "table_id": table_id,
        "page": page,
    }
    metadata.update(doc_instrument_meta)
    return Chunk(
        chunk_id=f"{doc.doc_id}-t{chunk_index}-{digest}",
        doc_id=doc.doc_id,
        text=table_text,
        start_char=start_char,
        end_char=end_char,
        metadata=metadata,
    )


def _strip_table_blocks(text: str) -> str:
    out = []
    cursor = 0
    for match in TABLE_BLOCK_RE.finditer(text):
        start, end = match.span()
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    merged = "".join(out)
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged.strip()


def chunk_document_with_table_awareness(
    doc: SourceDocument,
    target_chars: int = 700,
    overlap_chars: int = 120,
) -> List[Chunk]:
    table_matches = list(TABLE_BLOCK_RE.finditer(doc.text))
    doc_instrument_meta = _infer_document_instrument(doc.text, doc.title)

    text_doc = replace(doc, text=_strip_table_blocks(doc.text))
    text_chunks = semantic_chunk_document(
        text_doc,
        target_chars=target_chars,
        overlap_chars=overlap_chars,
        base_metadata={"content_type": "text"},
    )

    if not table_matches:
        return text_chunks

    table_chunks: List[Chunk] = []
    for idx, match in enumerate(table_matches):
        table_chunks.append(
            _make_table_chunk(
                doc=doc,
                table_id=match.group("table_id"),
                table_body=match.group("table"),
                chunk_index=idx,
                start_char=match.start(),
                end_char=match.end(),
                doc_instrument_meta=doc_instrument_meta,
            )
        )

    all_chunks = text_chunks + table_chunks
    all_chunks.sort(key=lambda chunk: (chunk.start_char, chunk.chunk_id))

    # Re-number chunk_index in metadata after sorting for deterministic order.
    normalized: List[Chunk] = []
    for idx, chunk in enumerate(all_chunks):
        metadata: Dict[str, str] = dict(chunk.metadata)
        metadata["chunk_index"] = str(idx)
        normalized.append(
            Chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata=metadata,
            )
        )
    return normalized

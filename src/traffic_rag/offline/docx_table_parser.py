import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    from docx import Document
    from docx.oxml.ns import qn
except ImportError:  # pragma: no cover
    Document = None
    qn = None


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_cell(cell: Optional[str]) -> str:
    if cell is None:
        return ""
    value = str(cell).replace("\x00", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|")


def _normalize_rows(rows: List[List[str]]) -> List[List[str]]:
    cleaned: List[List[str]] = []
    max_cols = 0
    for row in rows:
        out_row = [_clean_cell(cell) for cell in row]
        if not any(cell for cell in out_row):
            continue
        cleaned.append(out_row)
        max_cols = max(max_cols, len(out_row))

    if not cleaned:
        return []

    for idx, row in enumerate(cleaned):
        if len(row) < max_cols:
            cleaned[idx] = row + [""] * (max_cols - len(row))
    return cleaned


def _table_to_markdown(rows: List[List[str]]) -> str:
    normalized = _normalize_rows(rows)
    if not normalized:
        return ""
    if all(not any(cell for cell in row) for row in normalized):
        return ""

    first = normalized[0]
    non_empty_header = sum(1 for cell in first if cell)
    use_first_as_header = non_empty_header >= max(1, len(first) // 2)

    if use_first_as_header:
        header = [cell if cell else f"col_{i + 1}" for i, cell in enumerate(first)]
        body = normalized[1:]
    else:
        header = [f"col_{i + 1}" for i in range(len(first))]
        body = normalized

    body = [row for row in body if any(cell for cell in row)]
    if not body:
        body = [[""] * len(header)]

    lines = []
    lines.append("| " + " | ".join(_escape_md(cell) for cell in header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        lines.append("| " + " | ".join(_escape_md(cell) for cell in row) + " |")
    return "\n".join(lines)


def _is_usable_table(rows: List[List[str]]) -> bool:
    if not rows:
        return False
    if len(rows) < 2 or len(rows[0]) < 2:
        return False
    non_empty_per_row = [sum(1 for c in row if c) for row in rows]
    rich_rows = sum(1 for cnt in non_empty_per_row if cnt >= 2)
    total_non_empty = sum(non_empty_per_row)
    density = total_non_empty / float(len(rows) * len(rows[0]))
    return rich_rows >= 2 and density >= 0.25


def _extract_table_rows(table) -> List[List[str]]:
    """Extract rows from a python-docx Table, collapsing merged cells by reading cell text."""
    rows: List[List[str]] = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            cells.append(_clean_cell(cell.text))
        rows.append(cells)
    return rows


def _is_heading_paragraph(para) -> bool:
    """Return True if this paragraph looks like a section heading."""
    style_name = (para.style.name or "").lower() if para.style else ""
    if "heading" in style_name:
        return True
    text = para.text.strip()
    if not text:
        return False
    # Vietnamese legal section markers: Phần, Chương, Mục, Điều
    if re.match(r"^(Phần|Chương|Mục|Điều)\s+[IVXLCDM\d]+", text):
        return True
    return False


@dataclass(frozen=True)
class ExtractedTable:
    section: int
    table_index: int
    rows: int
    cols: int
    markdown: str


@dataclass(frozen=True)
class ExtractionSummary:
    docx_path: str
    text_out_path: str
    tables_out_path: str
    sections: int
    tables: int


def extract_docx_table_aware(
    docx_path: Path,
    text_out_path: Path,
    tables_out_path: Path,
) -> ExtractionSummary:
    if Document is None:
        raise RuntimeError(
            "python-docx is not installed. Install with: python3 -m pip install python-docx"
        )

    text_out_path.parent.mkdir(parents=True, exist_ok=True)
    tables_out_path.parent.mkdir(parents=True, exist_ok=True)

    doc = Document(str(docx_path))

    doc_lines: List[str] = []
    table_rows_jsonl: List[Dict] = []
    section = 1
    section_table_idx = 0
    total_tables = 0
    pending_text_lines: List[str] = []

    def flush_pending() -> None:
        block = _clean_text("\n".join(pending_text_lines))
        if block:
            doc_lines.append(block)
            doc_lines.append("")
        pending_text_lines.clear()

    # Walk the document body in order by inspecting the XML element tags
    # so paragraphs and tables appear interleaved in document order.
    body = doc.element.body
    para_tag = qn("w:p")
    table_tag = qn("w:tbl")

    for child in body:
        tag = child.tag

        if tag == para_tag:
            # Reconstruct a paragraph object to access .text and .style
            from docx.text.paragraph import Paragraph as _Para
            para = _Para(child, doc)
            text = para.text.strip()

            if _is_heading_paragraph(para):
                flush_pending()
                if section_table_idx > 0:
                    section += 1
                    section_table_idx = 0
                if text:
                    doc_lines.append(f"[SECTION {section}] {text}")
                    doc_lines.append("")
            elif text:
                pending_text_lines.append(text)

        elif tag == table_tag:
            flush_pending()

            from docx.table import Table as _Table
            tbl = _Table(child, doc)
            raw_rows = _extract_table_rows(tbl)
            normalized = _normalize_rows(raw_rows)

            if not _is_usable_table(normalized):
                continue

            markdown = _table_to_markdown(raw_rows)
            if not markdown:
                continue

            section_table_idx += 1
            total_tables += 1
            n_rows = len(normalized)
            n_cols = len(normalized[0]) if normalized else 0
            table_id = f"{section}.{section_table_idx}"

            doc_lines.append(f"[TABLE {table_id}] rows={n_rows} cols={n_cols}")
            doc_lines.append(markdown)
            doc_lines.append("")

            table_rows_jsonl.append(
                asdict(
                    ExtractedTable(
                        section=section,
                        table_index=section_table_idx,
                        rows=n_rows,
                        cols=n_cols,
                        markdown=markdown,
                    )
                )
            )

    flush_pending()

    text_out_path.write_text("\n".join(doc_lines).strip() + "\n", encoding="utf-8")
    with tables_out_path.open("w", encoding="utf-8") as f:
        for row in table_rows_jsonl:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return ExtractionSummary(
        docx_path=str(docx_path),
        text_out_path=str(text_out_path),
        tables_out_path=str(tables_out_path),
        sections=section,
        tables=total_tables,
    )

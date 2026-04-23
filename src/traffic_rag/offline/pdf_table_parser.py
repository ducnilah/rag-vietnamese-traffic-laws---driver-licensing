import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pdfplumber
except ImportError:  # pragma: no cover - handled at runtime for optional dependency.
    pdfplumber = None


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


def _normalize_rows(
    rows: List[List[Optional[str]]], drop_empty_rows: bool = True
) -> List[List[str]]:
    cleaned: List[List[str]] = []
    max_cols = 0
    for row in rows:
        out_row = [_clean_cell(cell) for cell in row]
        if drop_empty_rows and not any(cell for cell in out_row):
            continue
        cleaned.append(out_row)
        max_cols = max(max_cols, len(out_row))

    if not cleaned:
        return []

    for idx, row in enumerate(cleaned):
        if len(row) < max_cols:
            cleaned[idx] = row + [""] * (max_cols - len(row))
    return cleaned


def table_to_markdown(rows: List[List[Optional[str]]]) -> str:
    normalized_all = _normalize_rows(rows, drop_empty_rows=False)
    if not normalized_all:
        return ""

    if all(not any(cell for cell in row) for row in normalized_all):
        return ""

    first = normalized_all[0]
    non_empty_header_cells = sum(1 for cell in first if cell)
    use_first_as_header = non_empty_header_cells >= max(1, len(first) // 2)

    if use_first_as_header:
        header = [cell if cell else f"col_{i + 1}" for i, cell in enumerate(first)]
        body = normalized_all[1:]
    else:
        header = [f"col_{i + 1}" for i in range(len(first))]
        body = normalized_all

    body = [row for row in body if any(cell for cell in row)]
    if not body:
        body = [[""] * len(header)]

    lines = []
    lines.append("| " + " | ".join(_escape_md(cell) for cell in header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        lines.append("| " + " | ".join(_escape_md(cell) for cell in row) + " |")
    return "\n".join(lines)


@dataclass(frozen=True)
class ExtractedTable:
    page: int
    table_index: int
    rows: int
    cols: int
    markdown: str


@dataclass(frozen=True)
class ExtractionSummary:
    pdf_path: str
    text_out_path: str
    tables_out_path: str
    pages: int
    pages_with_tables: int
    tables: int


def _is_usable_table(rows: List[List[str]]) -> bool:
    if not rows:
        return False
    n_rows = len(rows)
    n_cols = len(rows[0])
    if n_rows < 2 or n_cols < 2:
        return False

    non_empty_per_row = [sum(1 for cell in row if cell) for row in rows]
    rich_rows = sum(1 for count in non_empty_per_row if count >= 2)
    total_non_empty = sum(non_empty_per_row)
    density = total_non_empty / float(n_rows * n_cols)

    if rich_rows < 2:
        return False
    if density < 0.25:
        return False
    return True


def extract_pdf_table_aware(
    pdf_path: Path,
    text_out_path: Path,
    tables_out_path: Path,
) -> ExtractionSummary:
    if pdfplumber is None:
        raise RuntimeError(
            "pdfplumber is not installed. Install with: python3 -m pip install --user pdfplumber"
        )

    text_out_path.parent.mkdir(parents=True, exist_ok=True)
    tables_out_path.parent.mkdir(parents=True, exist_ok=True)

    doc_lines: List[str] = []
    table_rows: List[Dict[str, object]] = []
    pages_with_tables = 0
    table_count = 0

    with pdfplumber.open(str(pdf_path)) as pdf:
        total_pages = len(pdf.pages)
        for page_no, page in enumerate(pdf.pages, start=1):
            page_text = _clean_text(page.extract_text() or "")
            doc_lines.append(f"[PAGE {page_no}]")
            doc_lines.append(page_text)

            raw_tables = page.extract_tables()
            table_blocks = []
            for table_idx, raw in enumerate(raw_tables, start=1):
                markdown = table_to_markdown(raw or [])
                if not markdown:
                    continue
                normalized = _normalize_rows(raw or [])
                if not _is_usable_table(normalized):
                    continue
                rows = len(normalized)
                cols = len(normalized[0]) if normalized else 0
                table_count += 1
                table_blocks.append((table_idx, markdown, rows, cols))

            if table_blocks:
                pages_with_tables += 1
                doc_lines.append("")
                doc_lines.append("### TABLES")
                for table_idx, markdown, rows, cols in table_blocks:
                    doc_lines.append(f"[TABLE {page_no}.{table_idx}] rows={rows} cols={cols}")
                    doc_lines.append(markdown)
                    doc_lines.append("")
                    table_rows.append(
                        asdict(
                            ExtractedTable(
                                page=page_no,
                                table_index=table_idx,
                                rows=rows,
                                cols=cols,
                                markdown=markdown,
                            )
                        )
                    )

            doc_lines.append("")

    text_out_path.write_text("\n".join(doc_lines).strip() + "\n", encoding="utf-8")
    with tables_out_path.open("w", encoding="utf-8") as f:
        for row in table_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return ExtractionSummary(
        pdf_path=str(pdf_path),
        text_out_path=str(text_out_path),
        tables_out_path=str(tables_out_path),
        pages=total_pages,
        pages_with_tables=pages_with_tables,
        tables=table_count,
    )

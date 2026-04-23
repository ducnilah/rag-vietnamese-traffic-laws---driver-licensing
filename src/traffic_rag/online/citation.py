from typing import Dict, List


def build_citation(metadata: Dict[str, str]) -> str:
    parts: List[str] = []

    instrument_ref = metadata.get("instrument_ref")
    if instrument_ref:
        parts.append(instrument_ref)
    else:
        instrument_type = metadata.get("instrument_type")
        instrument_no = metadata.get("instrument_no")
        if instrument_type and instrument_no:
            parts.append(f"{instrument_type} {instrument_no}")
        elif instrument_type:
            parts.append(instrument_type)

    article_no = metadata.get("article_no")
    chapter_no = metadata.get("chapter_no")
    part_no = metadata.get("part_no")
    table_id = metadata.get("table_id")
    page = metadata.get("page")

    if part_no:
        parts.append(f"Phần {part_no}")
    if chapter_no:
        parts.append(f"Chương {chapter_no}")
    if article_no:
        parts.append(f"Điều {article_no}")
    if table_id:
        parts.append(f"Bảng {table_id}")
    if page:
        parts.append(f"Trang {page}")

    source = metadata.get("source_path")
    if source:
        parts.append(f"nguồn: {source}")

    return " | ".join(parts) if parts else "nguồn nội bộ"

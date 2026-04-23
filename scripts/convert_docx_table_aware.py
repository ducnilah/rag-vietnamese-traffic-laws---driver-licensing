#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_rag.offline.docx_table_parser import extract_docx_table_aware


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DOCX to table-aware markdown artifacts."
    )
    parser.add_argument("--docx", type=Path, required=True, help="Input DOCX file")
    parser.add_argument(
        "--out-text",
        type=Path,
        default=None,
        help="Output markdown file. Default: data/processed/text/<stem>_table_aware.md",
    )
    parser.add_argument(
        "--out-tables",
        type=Path,
        default=None,
        help="Output tables JSONL file. Default: data/processed/text/<stem>_tables.jsonl",
    )
    args = parser.parse_args()

    if not args.docx.exists():
        raise FileNotFoundError(f"DOCX not found: {args.docx}")

    stem = args.docx.stem
    default_text = ROOT / "data" / "processed" / "text" / f"{stem}_table_aware.md"
    default_tables = ROOT / "data" / "processed" / "text" / f"{stem}_tables.jsonl"

    text_out = args.out_text if args.out_text else default_text
    tables_out = args.out_tables if args.out_tables else default_tables

    summary = extract_docx_table_aware(args.docx, text_out, tables_out)
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

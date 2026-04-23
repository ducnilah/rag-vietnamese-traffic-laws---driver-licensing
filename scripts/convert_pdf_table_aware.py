#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_rag.offline.pdf_table_parser import extract_pdf_table_aware


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDF to table-aware plaintext/markdown artifacts."
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Input PDF file")
    parser.add_argument(
        "--out-text",
        type=Path,
        default=None,
        help="Output text/markdown file. Default: data/processed/text/<pdf_stem>_table_aware.md",
    )
    parser.add_argument(
        "--out-tables",
        type=Path,
        default=None,
        help="Output tables JSONL file. Default: data/processed/text/<pdf_stem>_tables.jsonl",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    default_text = ROOT / "data" / "processed" / "text" / f"{args.pdf.stem}_table_aware.md"
    default_tables = ROOT / "data" / "processed" / "text" / f"{args.pdf.stem}_tables.jsonl"

    text_out = args.out_text if args.out_text else default_text
    tables_out = args.out_tables if args.out_tables else default_tables

    summary = extract_pdf_table_aware(args.pdf, text_out, tables_out)
    print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

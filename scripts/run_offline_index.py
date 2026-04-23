#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from traffic_rag.offline.indexing import OfflineIndexer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline indexing pipeline")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "txt_md",
        help="Primary input dir for .txt/.md",
    )
    parser.add_argument(
        "--extra-dir",
        type=Path,
        action="append",
        default=[ROOT / "data" / "processed" / "text"],
        help="Additional input dir(s), e.g. table-aware markdown outputs",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "index")
    parser.add_argument("--target-chars", type=int, default=1000)
    parser.add_argument("--overlap-chars", type=int, default=150)
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    indexer = OfflineIndexer(target_chars=args.target_chars, overlap_chars=args.overlap_chars)
    summary = indexer.build(args.raw_dir, args.out_dir, extra_dirs=args.extra_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

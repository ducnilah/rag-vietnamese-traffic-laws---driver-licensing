from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_law_v2.indexing import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v2 offline RAG index.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--index-dir", default="data/index")
    args = parser.parse_args()
    report = build_index(Path(args.raw_dir), Path(args.index_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

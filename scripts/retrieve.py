from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_law_v2.context import build_context
from traffic_law_v2.retrieval import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 retrieval smoke query.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context", action="store_true")
    args = parser.parse_args()
    hits = HybridRetriever(Path(args.index_dir)).retrieve(args.query, top_k=args.top_k)
    if args.context:
        print(json.dumps(build_context(args.query, hits).model_dump(), ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"hits": [hit.model_dump() for hit in hits]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_law_v2.context import build_context
from traffic_law_v2.generation import generate_answer
from traffic_law_v2.retrieval import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one end-to-end RAG answer.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    hits = HybridRetriever(Path(args.index_dir)).retrieve(args.query, top_k=args.top_k)
    context = build_context(args.query, hits)
    answer = generate_answer(args.query, context)
    print(json.dumps(answer.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

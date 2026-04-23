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

from traffic_rag.online.citation import build_citation
from traffic_rag.online.retrieval import has_table_intent, search_with_table_priority


def main() -> None:
    parser = argparse.ArgumentParser(description="Search index with table-priority reranking.")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data" / "index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    hits = search_with_table_priority(args.index_dir, args.query, top_k=args.top_k)
    output = {
        "query": args.query,
        "table_intent": has_table_intent(args.query),
        "hits": [
            {
                "chunk_id": hit.chunk_id,
                "content_type": hit.content_type,
                "raw_score": round(hit.raw_score, 5),
                "final_score": round(hit.final_score, 5),
                "table_id": hit.metadata.get("table_id"),
                "page": hit.metadata.get("page"),
                "citation": build_citation(hit.metadata),
                "preview": hit.text[:240],
            }
            for hit in hits
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

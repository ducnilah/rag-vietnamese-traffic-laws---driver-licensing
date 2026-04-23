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
from traffic_rag.online.retrieval import has_table_intent, search_hybrid, search_with_table_priority


def main() -> None:
    parser = argparse.ArgumentParser(description="Search index (sparse/hybrid) with citations.")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data" / "index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["sparse", "hybrid"],
        default="hybrid",
        help="Retrieval mode",
    )
    parser.add_argument(
        "--dense-backend",
        choices=["auto", "chroma", "jaccard"],
        default="auto",
        help="Dense backend for hybrid mode",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.mode == "sparse":
        hits_sparse = search_with_table_priority(args.index_dir, args.query, top_k=args.top_k)
        output_hits = [
            {
                "chunk_id": hit.chunk_id,
                "content_type": hit.content_type,
                "sparse_score": round(hit.raw_score, 5),
                "final_score": round(hit.final_score, 5),
                "table_id": hit.metadata.get("table_id"),
                "page": hit.metadata.get("page"),
                "citation": build_citation(hit.metadata),
                "preview": hit.text[:240],
            }
            for hit in hits_sparse
        ]
    else:
        hits_hybrid = search_hybrid(
            args.index_dir,
            args.query,
            top_k=args.top_k,
            dense_backend=args.dense_backend,
        )
        output_hits = [
            {
                "chunk_id": hit.chunk_id,
                "content_type": hit.content_type,
                "sparse_score": round(hit.sparse_score, 5),
                "dense_score": round(hit.dense_score, 5),
                "fused_score": round(hit.fused_score, 5),
                "final_score": round(hit.final_score, 5),
                "table_id": hit.metadata.get("table_id"),
                "page": hit.metadata.get("page"),
                "citation": hit.citation,
                "preview": hit.text[:240],
            }
            for hit in hits_hybrid
        ]

    output = {
        "query": args.query,
        "mode": args.mode,
        "dense_backend": args.dense_backend if args.mode == "hybrid" else None,
        "table_intent": has_table_intent(args.query),
        "hits": output_hits,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

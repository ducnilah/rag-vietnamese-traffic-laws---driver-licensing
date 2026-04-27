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

from traffic_rag.online.service import RetrievalService


def main() -> None:
    parser = argparse.ArgumentParser(description="Build M3 context package from retrieval results.")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data" / "index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--neighbor-window", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=1800)
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

    service = RetrievalService(args.index_dir, dense_backend=args.dense_backend)
    context = service.build_context(
        query=args.query,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        use_hybrid=args.mode == "hybrid",
        neighbor_window=args.neighbor_window,
        max_context_tokens=args.max_context_tokens,
    )

    output = {
        "query": context.query,
        "rewritten_query": context.rewritten_query,
        "mode": args.mode,
        "dense_backend": args.dense_backend if args.mode == "hybrid" else None,
        "estimated_tokens": context.estimated_tokens,
        "confidence": context.confidence,
        "chunks": [
            {
                "slot": chunk.slot,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "content_type": chunk.content_type,
                "score": round(chunk.final_score, 6),
                "citation": chunk.citation,
                "preview": chunk.text[:220],
            }
            for chunk in context.chunks
        ],
        "citation_map": context.citation_map,
        "context_text": context.context_text,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

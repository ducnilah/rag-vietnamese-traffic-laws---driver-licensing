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
from traffic_rag.vector.chroma import (
    CHROMA_COLLECTION_DEFAULT,
    CHROMA_DIRNAME_DEFAULT,
    CHROMA_EMBEDDING_BACKEND_DEFAULT,
    CHROMA_EMBEDDING_MODEL_DEFAULT,
)


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
    parser.add_argument(
        "--with-chroma",
        action="store_true",
        help="Also build ChromaDB dense index under out-dir/chroma",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=None,
        help=f"Override ChromaDB persist directory (default: <out-dir>/{CHROMA_DIRNAME_DEFAULT})",
    )
    parser.add_argument(
        "--chroma-collection",
        default=CHROMA_COLLECTION_DEFAULT,
        help=f"ChromaDB collection name (default: {CHROMA_COLLECTION_DEFAULT})",
    )
    parser.add_argument(
        "--chroma-embedding-backend",
        choices=["bge-m3", "hash"],
        default=CHROMA_EMBEDDING_BACKEND_DEFAULT,
        help=f"Embedding backend for Chroma index (default: {CHROMA_EMBEDDING_BACKEND_DEFAULT})",
    )
    parser.add_argument(
        "--chroma-embedding-model",
        default=CHROMA_EMBEDDING_MODEL_DEFAULT,
        help=f"Embedding model name for Chroma backend (default: {CHROMA_EMBEDDING_MODEL_DEFAULT})",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    indexer = OfflineIndexer(target_chars=args.target_chars, overlap_chars=args.overlap_chars)
    summary = indexer.build(
        args.raw_dir,
        args.out_dir,
        extra_dirs=args.extra_dir,
        build_chroma=args.with_chroma,
        chroma_dir=args.chroma_dir,
        chroma_collection=args.chroma_collection,
        chroma_embedding_backend=args.chroma_embedding_backend,
        chroma_embedding_model=args.chroma_embedding_model,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

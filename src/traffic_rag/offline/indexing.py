import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .bm25 import BM25Index
from .models import Chunk, SourceDocument
from .parser import parse_directory
from .quality import QualityReport, run_quality_checks
from .table_aware_chunker import chunk_document_with_table_awareness
from traffic_rag.vector.chroma import (
    CHROMA_COLLECTION_DEFAULT,
    CHROMA_DIRNAME_DEFAULT,
    CHROMA_EMBEDDING_BACKEND_DEFAULT,
    CHROMA_EMBEDDING_MODEL_DEFAULT,
    ChromaIndexer,
)

logger = logging.getLogger(__name__)


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class OfflineIndexer:
    def __init__(self, target_chars: int = 1000, overlap_chars: int = 150) -> None:
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def _load_documents(self, input_dirs: Iterable[Path]) -> List[SourceDocument]:
        docs: List[SourceDocument] = []
        seen_paths = set()
        for input_dir in input_dirs:
            if not input_dir.exists():
                continue
            try:
                parsed_docs = parse_directory(input_dir)
            except RuntimeError:
                continue
            for doc in parsed_docs:
                if doc.source_path in seen_paths:
                    continue
                seen_paths.add(doc.source_path)
                docs.append(doc)

        if not docs:
            joined = ", ".join(str(path) for path in input_dirs)
            raise RuntimeError(f"No supported documents found in: {joined}")
        return self._dedupe_prefer_table_aware(docs)

    def _dedupe_prefer_table_aware(self, docs: List[SourceDocument]) -> List[SourceDocument]:
        selected: Dict[str, SourceDocument] = {}
        selected_rank: Dict[str, int] = {}

        for doc in docs:
            stem = Path(doc.source_path).stem
            if stem.endswith("_table_aware"):
                key = stem[: -len("_table_aware")]
                rank = 2
            else:
                key = stem
                rank = 1

            prev_rank = selected_rank.get(key, -1)
            if rank >= prev_rank:
                selected[key] = doc
                selected_rank[key] = rank

        return sorted(selected.values(), key=lambda item: item.source_path)

    def build(
        self,
        raw_dir: Path,
        out_dir: Path,
        extra_dirs: Optional[List[Path]] = None,
        build_chroma: bool = False,
        chroma_dir: Optional[Path] = None,
        chroma_collection: str = CHROMA_COLLECTION_DEFAULT,
        chroma_embedding_backend: str = CHROMA_EMBEDDING_BACKEND_DEFAULT,
        chroma_embedding_model: str = CHROMA_EMBEDDING_MODEL_DEFAULT,
    ) -> Dict[str, object]:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("offline_index.build started out_dir=%s", out_dir)

        input_dirs = [raw_dir]
        if extra_dirs:
            input_dirs.extend(extra_dirs)
        docs: List[SourceDocument] = self._load_documents(input_dirs)
        logger.info("loaded_docs=%d input_dirs=%s", len(docs), [str(path) for path in input_dirs])

        chunks: List[Chunk] = []
        for doc in docs:
            before = len(chunks)
            chunks.extend(
                chunk_document_with_table_awareness(
                    doc,
                    target_chars=self.target_chars,
                    overlap_chars=self.overlap_chars,
                )
            )
            logger.debug(
                "doc_chunked doc_id=%s source=%s chunks_added=%d",
                doc.doc_id,
                doc.source_path,
                len(chunks) - before,
            )
        report: QualityReport = run_quality_checks(chunks)
        logger.info(
            "quality_checks total_chunks=%d warnings=%d errors=%d",
            report.total_chunks,
            report.warnings,
            report.errors,
        )

        if report.errors > 0:
            raise RuntimeError(
                f"Quality checks failed with {report.errors} errors. "
                "See quality_report.json for details."
            )

        bm25 = BM25Index()
        bm25.build(chunks)
        logger.info("bm25_built docs=%d", len(chunks))

        chroma_summary: Dict[str, object] = {"enabled": False}
        if build_chroma:
            vector_dir = chroma_dir if chroma_dir else out_dir / CHROMA_DIRNAME_DEFAULT
            chroma_indexer = ChromaIndexer(
                persist_dir=vector_dir,
                collection_name=chroma_collection,
                embedding_backend=chroma_embedding_backend,
                embedding_model=chroma_embedding_model,
            )
            chroma_summary = chroma_indexer.build(chunks)

        documents_path = out_dir / "documents.jsonl"
        chunks_path = out_dir / "chunks.jsonl"
        bm25_path = out_dir / "bm25.json"
        report_path = out_dir / "quality_report.json"

        _write_jsonl(documents_path, [asdict(doc) for doc in docs])
        _write_jsonl(chunks_path, [asdict(chunk) for chunk in chunks])
        bm25.save(bm25_path)
        logger.info("artifacts_written documents=%s chunks=%s bm25=%s", documents_path, chunks_path, bm25_path)

        chunk_type_counts: Dict[str, int] = {}
        for chunk in chunks:
            chunk_type = chunk.metadata.get("content_type", "unknown")
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1

        report_dict = {
            "total_chunks": report.total_chunks,
            "warnings": report.warnings,
            "errors": report.errors,
            "issues": [asdict(issue) for issue in report.issues],
        }
        report_path.write_text(json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("quality_report_written path=%s", report_path)

        return {
            "documents": len(docs),
            "chunks": len(chunks),
            "warnings": report.warnings,
            "errors": report.errors,
            "chunk_type_counts": chunk_type_counts,
            "input_dirs": [str(path) for path in input_dirs],
            "artifacts": {
                "documents": str(documents_path),
                "chunks": str(chunks_path),
                "bm25": str(bm25_path),
                "quality_report": str(report_path),
            },
            "chroma": chroma_summary,
        }

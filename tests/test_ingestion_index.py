from pathlib import Path

from traffic_law_v2.indexing import build_index, load_chunks
from traffic_law_v2.ingestion import load_documents


RAW_DIR = Path("data/raw")


def test_loads_user_docx_document() -> None:
    docs = load_documents(RAW_DIR)
    assert docs, "Expected at least one source document in data/raw"
    assert any(doc.metadata.source_path.endswith("xu_phat_long.docx") for doc in docs)
    assert any("phạt" in doc.text.lower() or "xử phạt" in doc.text.lower() for doc in docs)


def test_build_index_writes_chunks(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    report = build_index(RAW_DIR, index_dir)
    chunks = load_chunks(index_dir)
    assert report["documents"] >= 1
    assert report["chunks"] == len(chunks)
    assert chunks
    assert (index_dir / "documents.jsonl").exists()
    assert (index_dir / "chunks.jsonl").exists()
    assert (index_dir / "bm25.json").exists()
    assert (index_dir / "quality_report.json").exists()

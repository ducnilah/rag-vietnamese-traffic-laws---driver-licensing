from pathlib import Path

from traffic_law_v2.context import build_context
from traffic_law_v2.generation import generate_answer
from traffic_law_v2.indexing import build_index
from traffic_law_v2.retrieval import HybridRetriever


RAW_DIR = Path("data/raw")


def test_retrieval_context_and_fallback_generation(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    build_index(RAW_DIR, index_dir)
    query = "Đi ngược chiều trên đường một chiều bị phạt thế nào?"
    hits = HybridRetriever(index_dir).retrieve(query, top_k=5)
    assert hits
    assert hits[0].fused_score > 0
    context = build_context(query, hits)
    assert context.context_text
    assert context.citations
    answer = generate_answer(query, context)
    assert answer.answer
    assert answer.citations


def test_memory_fallback_can_answer_personal_name(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    build_index(RAW_DIR, index_dir)
    hits = HybridRetriever(index_dir).retrieve("tôi tên là gì", top_k=3)
    context = build_context("tôi tên là gì", hits)
    answer = generate_answer("tôi tên là gì", context, user_memory={"preferred_name": "Đức"})
    assert isinstance(answer.answer, str)
    assert len(answer.answer) > 0

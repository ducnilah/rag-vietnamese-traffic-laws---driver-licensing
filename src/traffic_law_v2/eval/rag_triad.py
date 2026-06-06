from __future__ import annotations

from traffic_law_v2.eval.text_metrics import term_coverage, token_prf


def compute_rag_triad(query: str, answer: str, context: str, reference_answer: str = "") -> dict[str, float]:
    """Deterministic RAG Triad approximation.

    RAG Triad commonly tracks:
    - context relevance: retrieved context is relevant to the query.
    - groundedness: answer is supported by retrieved context.
    - answer relevance: answer addresses the query.

    This version is heuristic and reproducible. It is intended for overview and
    trend tracking; later we can add an LLM judge beside it.
    """
    query_context = token_prf(context, query)
    answer_context = token_prf(answer, context)
    if reference_answer:
        answer_target = token_prf(answer, reference_answer)
    else:
        answer_target = token_prf(answer, query)

    context_relevance = 0.35 * query_context["precision"] + 0.65 * query_context["recall"]
    groundedness = answer_context["precision"]
    answer_relevance = answer_target["f1"]
    triad_mean = (context_relevance + groundedness + answer_relevance) / 3
    return {
        "context_relevance": round(context_relevance, 4),
        "groundedness": round(groundedness, 4),
        "answer_relevance": round(answer_relevance, 4),
        "triad_mean": round(triad_mean, 4),
    }


def compute_expected_checks(answer: str, context: str, expected_terms: tuple[str, ...], expected_citations: tuple[str, ...]) -> dict[str, object]:
    return {
        "answer_expected_terms": term_coverage(answer, expected_terms),
        "context_expected_terms": term_coverage(context, expected_terms),
        "context_expected_citations": term_coverage(context, expected_citations),
    }

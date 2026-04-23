from typing import Dict, Iterable


def minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Dict[str, int]],
    k: int = 60,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for rank_map in ranked_lists:
        for chunk_id, rank in rank_map.items():
            out[chunk_id] = out.get(chunk_id, 0.0) + 1.0 / float(k + rank)
    return out


def weighted_hybrid_score(
    sparse_norm: float,
    dense_norm: float,
    rrf_score: float,
    sparse_weight: float = 0.55,
    dense_weight: float = 0.35,
    rrf_weight: float = 0.10,
) -> float:
    return (
        sparse_weight * sparse_norm
        + dense_weight * dense_norm
        + rrf_weight * rrf_score
    )

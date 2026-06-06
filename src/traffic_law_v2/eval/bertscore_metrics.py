from __future__ import annotations

from typing import Any


def compute_bertscore(candidates: list[str], references: list[str], model_type: str = "bert-base-multilingual-cased") -> dict[str, Any]:
    """Compute BERTScore P/R/F1 if the optional bert-score package is installed.

    The project keeps this optional because BERTScore pulls a transformer model
    and can be slow/heavy on local machines. The script reports availability
    explicitly instead of pretending a fallback score is real BERTScore.
    """
    try:
        from bert_score import score
    except Exception as exc:  # pragma: no cover - depends on optional package
        return {
            "available": False,
            "error": f"bert-score is not installed or failed to import: {type(exc).__name__}: {exc}",
            "model_type": model_type,
            "items": [],
        }

    try:
        precision, recall, f1 = score(
            candidates,
            references,
            lang="vi",
            model_type=model_type,
            verbose=False,
            rescale_with_baseline=False,
        )
    except Exception as exc:  # pragma: no cover - depends on local HF cache/network
        return {
            "available": False,
            "error": f"BERTScore model failed to load or run: {type(exc).__name__}: {exc}",
            "model_type": model_type,
            "items": [],
        }
    items = []
    for p, r, f in zip(precision.tolist(), recall.tolist(), f1.tolist()):
        items.append({"precision": round(float(p), 6), "recall": round(float(r), 6), "f1": round(float(f), 6)})
    return {"available": True, "error": None, "model_type": model_type, "items": items}

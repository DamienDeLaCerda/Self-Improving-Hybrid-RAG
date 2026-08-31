"""
Cross-encoder re-ranking for hybrid retrieval results.

Ranks (query, document) pairs so the most relevant chunks
are passed to the generator first.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config.settings import RERANKER_MODEL_NAME

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_RERANKER = _PROJECT_ROOT / "models" / "ms-marco-MiniLM-L-6-v2"


def _model_source() -> str:
    if _LOCAL_RERANKER.exists():
        return str(_LOCAL_RERANKER)
    return RERANKER_MODEL_NAME


@lru_cache(maxsize=1)
def _load_cross_encoder():
    """Load CrossEncoder once; return None if unavailable."""
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(_model_source())
        print(f"Reranker ready: {_model_source()}")
        return model
    except Exception as exc:
        print(f"Reranker unavailable ({exc}); using lexical fallback.")
        return None


def _tokenize(text: str) -> list[str]:
    return [t.strip(".,!?;:()[]\"'").lower() for t in text.split() if t.strip(".,!?;:()[]\"'")]


def _lexical_score(query: str, doc: str) -> float:
    """Simple token-overlap score used when CrossEncoder is offline."""
    q_tokens = set(_tokenize(query))
    d_tokens = _tokenize(doc)
    if not q_tokens or not d_tokens:
        return 0.0
    hits = sum(1 for t in d_tokens if t in q_tokens)
    return hits / (len(d_tokens) ** 0.5)


def rerank(query: str, documents: list[str], top_k: int = 5) -> list[str]:
    """
    Re-rank retrieved documents for a query.

    Uses a cross-encoder when available; otherwise falls back to
    lexical overlap so the pipeline still runs offline.
    """
    if not documents:
        return []

    top_k = max(1, min(top_k, len(documents)))
    model = _load_cross_encoder()

    if model is None:
        scored = [(_lexical_score(query, doc), doc) for doc in documents]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)
    return [doc for _, doc in ranked[:top_k]]

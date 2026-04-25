from __future__ import annotations

import logging

from models.schemas import QueryType
from services.retriever import embed_query

logger = logging.getLogger("ai_research_assistant.router")

# Anchor descriptions that define each intent category.
# embed_query is called lazily so the embedding model loads only once.
INTENT_ANCHORS: dict[str, str] = {
    "research": "explain in depth how something works, detailed exploration of a topic",
    "summary": "summarize this document, give me a brief overview",
    "qa": "what is the answer to this specific factual question",
    "reasoning": "compare these two things, analyze trade-offs and differences",
}

_INTENT_VECTORS: dict[str, list[float]] = {}


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a * norm_b else 0.0


def _ensure_intent_vectors() -> dict[str, list[float]]:
    global _INTENT_VECTORS
    if not _INTENT_VECTORS:
        _INTENT_VECTORS = {k: embed_query(v) for k, v in INTENT_ANCHORS.items()}
    return _INTENT_VECTORS


def classify_query(query: str) -> QueryType:
    """Zero-cost intent classification using embedding cosine similarity.

    Eliminates one LLM round-trip per query by reusing the embedding model
    already loaded for retrieval.
    """
    intent_vecs = _ensure_intent_vectors()
    qvec = embed_query(query)
    scores = {k: _cosine_sim(qvec, v) for k, v in intent_vecs.items()}
    best = max(scores, key=scores.get)
    logger.info("router.classify", extra={"result": best, "scores": {k: round(v, 3) for k, v in scores.items()}})
    return QueryType(best)

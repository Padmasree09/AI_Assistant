from __future__ import annotations

import hashlib
import math
import re

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from core.config import get_settings

client: QdrantClient | None = None
embedder: SentenceTransformer | None = None

# bge requires this prefix on queries (not on documents)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _get_client() -> QdrantClient:
    global client
    if client is None:
        settings = get_settings()
        if settings.retriever_mode == "local":
            client = QdrantClient(path=settings.qdrant_local_path)
        else:
            client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    return client


def _get_embedder() -> SentenceTransformer:
    global embedder
    if embedder is None:
        settings = get_settings()
        embedder = SentenceTransformer(settings.embedding_model_name, local_files_only=True)
    return embedder


def _hash_embed(text: str) -> list[float]:
    settings = get_settings()
    size = settings.embedding_size
    vector = [0.0] * size
    tokens = re.findall(r"\w+", text.lower())

    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % size
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def embed_query(query: str) -> list[float]:
    settings = get_settings()
    if settings.embedding_backend == "sentence-transformers":
        prefixed = BGE_QUERY_PREFIX + query
        return _get_embedder().encode(prefixed, normalize_embeddings=True).tolist()
    return _hash_embed(query)


def embed_document(text: str) -> list[float]:
    settings = get_settings()
    if settings.embedding_backend == "sentence-transformers":
        return _get_embedder().encode(text, normalize_embeddings=True).tolist()
    return _hash_embed(text)


def ensure_collection() -> None:
    settings = get_settings()
    active_client = _get_client()

    if active_client.collection_exists(settings.qdrant_collection):
        return

    vector_size = len(embed_query("collection bootstrap"))
    active_client.create_collection(
        collection_name=settings.qdrant_collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def index_documents(documents: list[dict]) -> int:
    if not documents:
        return 0

    settings = get_settings()
    ensure_collection()

    points = []
    for idx, doc in enumerate(documents, start=1):
        text = doc.get("text", "").strip()
        if not text:
            continue

        points.append(
            PointStruct(
                id=idx,
                vector=embed_document(text),
                payload={"text": text, "source": doc.get("source", f"doc-{idx}")},
            )
        )

    if not points:
        return 0

    _get_client().upsert(collection_name=settings.qdrant_collection, points=points)
    return len(points)


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    settings = get_settings()
    vector = embed_query(query)
    try:
        response = _get_client().query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            limit=top_k,
        )
    except (ValueError, UnexpectedResponse):
        return []

    results = response.points
    return [
        {
            "text": (r.payload.get("text") or r.payload.get("text:") or ""),
            "source": r.payload.get("source", "unknown"),
            "score": r.score or 0.0,
        }
        for r in results
    ]

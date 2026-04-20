from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
COLLECTION = "knowledge_base"

# bge requires this prefix on queries (not on documents)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_query(query: str) -> list[float]:
    prefixed = BGE_QUERY_PREFIX + query
    return embedder.encode(prefixed, normalize_embeddings=True).tolist()


def embed_document(text: str) -> list[float]:
    return embedder.encode(text, normalize_embeddings=True).tolist()


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    vector = embed_query(query)
    results = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=top_k,
    )
    return [
        {
            "text": (r.payload.get("text") or r.payload.get("text:") or ""),
            "source": r.payload.get("source", "unknown"),
            "score": r.score,
        }
        for r in results
    ]
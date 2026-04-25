from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[str]:
    """Split *text* into word-level chunks with overlap.

    Designed for phi-3 mini 4k: ~200-word chunks let you fit 3-5 chunks
    plus the prompt and output budget inside the 4096-token window.

    Parameters
    ----------
    text:
        Raw document text.
    chunk_size:
        Maximum number of words per chunk.
    overlap:
        Number of words to repeat between consecutive chunks so that
        context is preserved across boundaries.

    Returns
    -------
    list[str]
        One or more text chunks.  Short texts are returned as-is.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

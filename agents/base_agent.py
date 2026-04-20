from __future__ import annotations

from services.llm import call_llm
from services.retriever import retrieve


class BaseAgent:
    max_tool_iterations = 2

    def build_prompt(self, query: str, context: str) -> str:
        raise NotImplementedError

    def run(self, query: str, top_k: int = 5) -> dict:
        return self.run_with_tools(query=query, top_k=top_k)

    def run_with_tools(self, query: str, top_k: int = 5) -> dict:
        """Tool-style execution loop: retrieve, refine, retrieve, then answer."""
        gathered_chunks: list[dict] = []
        retrieval_query = query

        for _ in range(self.max_tool_iterations):
            chunks = retrieve(retrieval_query, top_k)
            if chunks:
                gathered_chunks.extend(chunks)

            if not self._should_refine(chunks):
                break

            retrieval_query = self._refine_query(query, chunks)

        # de-duplicate chunks by (source, text)
        seen = set()
        unique_chunks = []
        for chunk in gathered_chunks:
            key = (chunk.get("source", "unknown"), chunk.get("text", ""))
            if key in seen:
                continue
            seen.add(key)
            unique_chunks.append(chunk)

        context = "\n\n".join(c.get("text", "") for c in unique_chunks)
        sources = list(dict.fromkeys(c.get("source", "unknown") for c in unique_chunks))

        prompt = self.build_prompt(query, context)
        answer = call_llm(prompt)

        return {"answer": answer, "sources": sources}

    def _should_refine(self, chunks: list[dict]) -> bool:
        if not chunks:
            return True
        avg_score = sum(c.get("score", 0.0) for c in chunks) / len(chunks)
        return avg_score < 0.35

    def _refine_query(self, original_query: str, chunks: list[dict]) -> str:
        sample_context = "\n".join(c.get("text", "")[:180] for c in chunks[:2])
        prompt = f"""Rewrite the search query to improve retrieval quality.
Return only one short rewritten query.

Original query: {original_query}
Retrieved snippets:
{sample_context or '(none)'}
"""
        try:
            rewritten = call_llm(prompt, max_tokens=40).strip()
            return rewritten if rewritten else original_query
        except Exception:
            return original_query
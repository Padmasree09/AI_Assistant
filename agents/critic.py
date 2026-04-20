from __future__ import annotations

import json
import logging

from services.llm import call_llm


class Critic:
    """Evaluates answer quality and recommends regeneration when needed."""

    def __init__(self, min_score: int = 7) -> None:
        self.min_score = min_score
        self.logger = logging.getLogger("ai_research_assistant.critic")

    def review(self, query: str, answer: str, context: str) -> dict:
        prompt = f"""You are an answer critic.
Evaluate the answer on:
1) Grounding in context
2) Relevance to query
3) Completeness

Return STRICT JSON only with keys:
score (0-10 integer), needs_revision (boolean), feedback (string).

Query: {query}
Context:
{context[:2500]}

Answer:
{answer}
"""
        try:
            raw = call_llm(prompt, max_tokens=140)
            parsed = json.loads(raw)
            score = int(parsed.get("score", 0))
            needs_revision = bool(parsed.get("needs_revision", score < self.min_score))
            feedback = str(parsed.get("feedback", "Improve factual grounding and coverage."))
            result = {"score": score, "needs_revision": needs_revision or score < self.min_score, "feedback": feedback}
            self.logger.info("critic.review", extra={"score": score, "needs_revision": result["needs_revision"]})
            return result
        except Exception as exc:  # pragma: no cover
            self.logger.warning("critic.fallback", extra={"error": str(exc)})
            too_short = len(answer.split()) < 40
            no_context = not context.strip()
            needs_revision = too_short or no_context
            return {
                "score": 5 if needs_revision else 8,
                "needs_revision": needs_revision,
                "feedback": "Answer is too short or weakly grounded. Expand with concrete evidence from retrieved context.",
            }
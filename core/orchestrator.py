from __future__ import annotations

import logging
import time
from collections import OrderedDict

from agents.critic import Critic
from agents.planner import Planner
from models.schemas import QueryType
from router.query_router import classify_query
from services.cache import ResponseCache
from core.config import get_settings
from services.llm import call_llm
from services.memory import SessionMemory


class Orchestrator:
    """Central brain coordinating planning, tool-based execution, critique, and memory."""

    def __init__(
        self,
        agent_registry: dict[QueryType, object],
        planner: Planner | None = None,
        critic: Critic | None = None,
        memory: SessionMemory | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.agent_registry = agent_registry
        self.planner = planner or Planner()
        self.critic = critic or Critic()
        self.memory = memory or SessionMemory()
        self.cache = cache or ResponseCache()
        self.logger = logging.getLogger("ai_research_assistant.orchestrator")

    def handle_query(self, query: str, top_k: int = 5, session_id: str = "default") -> dict:
        t0 = time.perf_counter()
        query_type = classify_query(query)
        self.logger.info("orchestrator.query_type", extra={"query_type": query_type.value})

        cache_key = self.cache.build_key(query, top_k)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.logger.info("orchestrator.cache_hit", extra={"session_id": session_id})
            return {
                "query_type": query_type,
                "answer": cached["answer"],
                "sources": cached["sources"],
                "meta": {"cached": True, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)},
            }

        history = self.memory.format_history(session_id=session_id, limit=3)
        working_query = query
        if history:
            working_query = (
                "Use the recent conversation context when relevant.\n"
                f"Conversation:\n{history}\n\n"
                f"Current user question: {query}"
            )

        plan = self.planner.create_plan(working_query, query_type)
        step_outputs = []
        source_bag: OrderedDict[str, None] = OrderedDict()

        for step in plan:
            step_start = time.perf_counter()
            agent = self.agent_registry.get(step.agent_type, self.agent_registry[query_type])

            step_prompt = f"Step objective: {step.description}\n\nUser query with context: {working_query}"
            result = agent.run(step_prompt, top_k=top_k)

            for src in result.get("sources", []):
                source_bag[src] = None

            step_latency_ms = round((time.perf_counter() - step_start) * 1000, 2)
            self.logger.info(
                "orchestrator.step_complete",
                extra={
                    "step_id": step.id,
                    "agent": step.agent_type.value,
                    "latency_ms": step_latency_ms,
                    "retrieval_count": len(result.get("sources", [])),
                },
            )
            step_outputs.append(
                {
                    "step": step.description,
                    "agent": step.agent_type.value,
                    "answer": result.get("answer", ""),
                }
            )

        synthesized_answer = self._synthesize_answer(query=query, query_type=query_type, step_outputs=step_outputs, history=history)

        evidence_context = "\n\n".join(output["answer"] for output in step_outputs)
        # Only run the expensive critic for multi-step plans or reasoning queries
        # to save LLM calls on simple QA / summary tasks.
        should_critique = (
            get_settings().enable_critic
            and (len(plan) > 1 or query_type == QueryType.REASONING)
        )
        if should_critique:
            critique = self.critic.review(query=query, answer=synthesized_answer, context=evidence_context)
        else:
            critique = {"score": None, "needs_revision": False, "feedback": "Critic skipped (simple query)."}

        if critique["needs_revision"]:
            synthesized_answer = self._revise_answer(query=query, query_type=query_type, current_answer=synthesized_answer, feedback=critique["feedback"], evidence=evidence_context)

        final_sources = list(source_bag.keys())
        self.memory.add_turn(session_id=session_id, query=query, answer=synthesized_answer)
        self.cache.set(cache_key, {"answer": synthesized_answer, "sources": final_sources})

        total_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        self.logger.info("orchestrator.complete", extra={"latency_ms": total_latency_ms, "steps": len(plan)})

        return {
            "query_type": query_type,
            "answer": synthesized_answer,
            "sources": final_sources,
            "meta": {"cached": False, "latency_ms": total_latency_ms, "steps": len(plan), "critique": critique},
        }

    def _synthesize_answer(self, query: str, query_type: QueryType, step_outputs: list[dict], history: str) -> str:
        reasoning_trace = "\n\n".join(
            f"Evidence {idx}:\nTask: {item['step']}\nAgent: {item['agent']}\nContent: {item['answer']}"
            for idx, item in enumerate(step_outputs, start=1)
        )
        prompt = f"""You are the final response composer for an offline AI assistant.
Write one clean final answer using only the evidence below.

Query type: {query_type.value}
User query: {query}
Conversation context:
{history or "(none)"}

Evidence:
{reasoning_trace}

Requirements:
- Answer the user query directly
- Keep the answer clean and presentable
- Do not repeat prompt sections or metadata
- Do not mention "query type", "conversation context", "step outputs", or "evidence"
- Prefer 1-3 short paragraphs unless bullets clearly fit better
- If evidence is weak, clearly state uncertainty in one sentence

Return only the final answer text.
"""
        return call_llm(prompt, max_tokens=get_settings().synthesis_max_tokens)

    def _revise_answer(self, query: str, query_type: QueryType, current_answer: str, feedback: str, evidence: str) -> str:
        prompt = f"""Improve the answer based on critic feedback.

Query type: {query_type.value}
User query: {query}
Critic feedback: {feedback}
Current answer:
{current_answer}

Evidence:
{evidence[:3000]}

Rules:
- Keep the answer direct and polished
- Remove prompt leakage or repeated metadata
- Stay grounded in the evidence

Return only the improved final answer text.
"""
        return call_llm(prompt, max_tokens=get_settings().synthesis_max_tokens)

    # ------------------------------------------------------------------
    # Streaming helper: runs everything except the final synthesis call
    # ------------------------------------------------------------------
    def prepare_query(self, query: str, top_k: int = 5, session_id: str = "default") -> dict:
        """Run route -> plan -> agent execution and return the synthesis prompt.

        Used by the ``/query/stream`` SSE endpoint so it can stream the
        final synthesis tokens instead of waiting for the full answer.
        """
        query_type = classify_query(query)
        history = self.memory.format_history(session_id=session_id, limit=3)
        working_query = query
        if history:
            working_query = (
                "Use the recent conversation context when relevant.\n"
                f"Conversation:\n{history}\n\n"
                f"Current user question: {query}"
            )

        plan = self.planner.create_plan(working_query, query_type)
        step_outputs = []
        source_bag: OrderedDict[str, None] = OrderedDict()

        for step in plan:
            agent = self.agent_registry.get(step.agent_type, self.agent_registry[query_type])
            step_prompt = f"Step objective: {step.description}\n\nUser query with context: {working_query}"
            result = agent.run(step_prompt, top_k=top_k)
            for src in result.get("sources", []):
                source_bag[src] = None
            step_outputs.append(
                {"step": step.description, "agent": step.agent_type.value, "answer": result.get("answer", "")}
            )

        synthesis_prompt = self._build_synthesis_prompt(
            query=query, query_type=query_type, step_outputs=step_outputs, history=history
        )
        return {"synthesis_prompt": synthesis_prompt, "sources": list(source_bag.keys()), "query_type": query_type}

    def _build_synthesis_prompt(self, query: str, query_type: QueryType, step_outputs: list[dict], history: str) -> str:
        reasoning_trace = "\n\n".join(
            f"Evidence {idx}:\nTask: {item['step']}\nAgent: {item['agent']}\nContent: {item['answer']}"
            for idx, item in enumerate(step_outputs, start=1)
        )
        return f"""You are the final response composer for an offline AI assistant.
Write one clean final answer using only the evidence below.

Query type: {query_type.value}
User query: {query}
Conversation context:
{history or "(none)"}

Evidence:
{reasoning_trace}

Requirements:
- Answer the user query directly
- Keep the answer clean and presentable
- Do not repeat prompt sections or metadata
- Do not mention "query type", "conversation context", "step outputs", or "evidence"
- Prefer 1-3 short paragraphs unless bullets clearly fit better
- If evidence is weak, clearly state uncertainty in one sentence

Return only the final answer text.
"""

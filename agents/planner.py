from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from models.schemas import QueryType
from services.llm import call_llm


@dataclass
class PlanStep:
    id: int
    description: str
    agent_type: QueryType


class Planner:
    """Builds a lightweight execution plan for a user query."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("ai_research_assistant.planner")

    def create_plan(self, query: str, query_type: QueryType) -> list[PlanStep]:
        heuristic_plan = self._heuristic_plan(query, query_type)
        llm_plan = self._llm_plan(query, query_type)
        if llm_plan:
            self.logger.info("planner.generated_plan", extra={"steps": len(llm_plan), "mode": "llm"})
            return llm_plan

        self.logger.info("planner.generated_plan", extra={"steps": len(heuristic_plan), "mode": "heuristic"})
        return heuristic_plan

    def _llm_plan(self, query: str, query_type: QueryType) -> list[PlanStep]:
        prompt = f"""You are a planning module for an offline AI research assistant.
Break the query into 2-4 concrete steps.
Return ONLY lines in this format:
STEP: <one short step>

Allowed agent types: research, summary, qa, reasoning.
Append " | agent=<type>" to each line.

User query: {query}
Default type: {query_type.value}
"""
        try:
            raw = call_llm(prompt, max_tokens=180)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("planner.llm_failure", extra={"error": str(exc)})
            return []

        steps: list[PlanStep] = []
        for idx, line in enumerate(raw.splitlines(), start=1):
            clean = line.strip()
            if not clean.lower().startswith("step:"):
                continue

            body = clean.split(":", 1)[1].strip()
            desc = body
            agent = query_type

            m = re.search(r"\|\s*agent\s*=\s*(research|summary|qa|reasoning)", body, flags=re.IGNORECASE)
            if m:
                desc = re.sub(r"\|\s*agent\s*=\s*(research|summary|qa|reasoning)", "", body, flags=re.IGNORECASE).strip()
                agent = QueryType(m.group(1).lower())

            if desc:
                steps.append(PlanStep(id=len(steps) + 1, description=desc, agent_type=agent))

        return steps[:4]

    def _heuristic_plan(self, query: str, query_type: QueryType) -> list[PlanStep]:
        q = query.lower()
        if any(token in q for token in ["compare", "vs", "versus", "difference"]):
            parts = re.split(r"\bvs\b|\bversus\b|\band\b", query, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                left = parts[0].replace("compare", "").strip(" :,-")
                right = parts[1].strip(" :,-")
                if left and right:
                    return [
                        PlanStep(1, f"Retrieve core information about {left}", QueryType.RESEARCH),
                        PlanStep(2, f"Retrieve core information about {right}", QueryType.RESEARCH),
                        PlanStep(3, f"Compare {left} and {right} with strengths and limitations", QueryType.REASONING),
                    ]

        return [
            PlanStep(1, f"Gather relevant evidence for: {query}", QueryType.RESEARCH if query_type == QueryType.REASONING else query_type),
            PlanStep(2, "Synthesize a final answer grounded in retrieved evidence", query_type),
        ]
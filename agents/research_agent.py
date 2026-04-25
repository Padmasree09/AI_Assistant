from agents.base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    def build_prompt(self, query: str, context: str) -> str:
        return f"""You are a research assistant.
Use only the evidence below to answer the user query.

Rules:
- Write a clear, well-structured explanation in 1-3 short paragraphs
- Do not repeat the prompt
- Do not include headings like "Answer" or "Final Answer"
- If the evidence is weak, say so briefly

User query:
{query}

Evidence:
{context or "(no evidence retrieved)"}

Return only the answer text.
"""

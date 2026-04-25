from agents.base_agent import BaseAgent


class SummaryAgent(BaseAgent):
    def build_prompt(self, query: str, context: str) -> str:
        return f"""You are a summarization assistant.
Summarize the evidence below for the user request.

Rules:
- Use 3-5 concise bullet points
- Do not repeat the request verbatim
- Do not include labels like "Summary"
- Stay grounded in the evidence

User request:
{query}

Evidence:
{context or "(no evidence retrieved)"}

Return only the bullet list.
"""

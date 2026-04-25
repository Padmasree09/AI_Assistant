from agents.base_agent import BaseAgent


class ReasoningAgent(BaseAgent):
    def build_prompt(self, query: str, context: str) -> str:
        return f"""You are an analytical assistant.
Provide a short reasoned analysis based only on the evidence below.

Rules:
- Focus on comparisons, trade-offs, or explanation of reasoning
- Keep it concise and readable
- Do not expose chain-of-thought
- Do not include labels like "Analysis"
- If evidence is limited, mention uncertainty briefly

Question:
{query}

Evidence:
{context or "(no evidence retrieved)"}

Return only the final analysis text.
"""

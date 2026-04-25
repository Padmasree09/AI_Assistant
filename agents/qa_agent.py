from agents.base_agent import BaseAgent


class QAAgent(BaseAgent):
    def build_prompt(self, query: str, context: str) -> str:
        return f"""You are a precise QA assistant.
Answer the question directly using only the evidence below.

Rules:
- Keep the answer short and clear
- Use at most 4 sentences
- Do not repeat the question
- Do not include labels like "Answer"
- If the evidence is insufficient, say that plainly

Question:
{query}

Evidence:
{context or "(no evidence retrieved)"}

Return only the answer text.
"""

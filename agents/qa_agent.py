from agents.base_agent import BaseAgent

class QAAgent(BaseAgent):
    def build_prompt(self, query: str, context: str) -> str:
        return f"""You are a precise QA assistant. Answer the question directly using only the context provided. Be concise.

        Context:
        {context}

        Question: {query} 
        Answer: """
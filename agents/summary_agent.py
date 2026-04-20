from agents.base_agent import BaseAgent

class SummaryAgent(BaseAgent):
    def build_prompt(self, query: str, context: str) -> str:
        return f"""You are a summarization assistant. Summarize the following context clearly and concisely in bullet points.
        Context:
        {context}
        Request: {query} 
        Summary: """
        
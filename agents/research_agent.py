from agents.base_agent import BaseAgent

class ResearchAgent(BaseAgent):
    def build_prompt(self, query: str, context: str) -> str:
        return f"""You are a research assistant. Using the context below, provide a thorough, well-structured explanation of the topic.
        Context:
        {context}
        Question: {query}
        Answer: 
        
        """
    
    
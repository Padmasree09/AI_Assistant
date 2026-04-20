from agents.base_agent import BaseAgent

class ReasoningAgent(BaseAgent):
    def build_prompt(self, query: str, context: str) -> str:
        return f"""You are an analytical assistant. Think step by step and provide a reasoned comparision or analysis based on the context.

        Context: 
        {context}
        
        Question : {query}
        Analysis: """
    
    
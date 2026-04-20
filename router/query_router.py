from services.llm import call_llm
from models.schemas import QueryType

ROUTER_PROMPT = """You are a query classifier. Given a user query, classify it into exactly one of these types:
- research : User wants to learn or explore a topic in depth
- summary  : User wants a document or topic summarized
- qa       : User has a direct factual question
- reasoning: User wants comparision, analysis, or reasoning

Reply with ONLY one word - the type. Nothing else
Query: {query}
Type:"""

def classify_query(query: str)-> QueryType:
    prompt = ROUTER_PROMPT.format(query=query)
    result = call_llm(prompt, max_tokens=5).strip().lower()
    
    #fallback to QA if model returns unexpected output

    type_map = {
        "research" : QueryType.RESEARCH,
        "summary"  : QueryType.SUMMARY,
        "qa"       : QueryType.QA,
        "reasoning": QueryType.REASONING
    }
    return type_map.get(result, QueryType.QA)

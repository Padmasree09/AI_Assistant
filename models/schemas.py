from enum import Enum

from pydantic import BaseModel


class QueryType(str, Enum):
    RESEARCH = "research"
    SUMMARY = "summary"
    QA = "qa"
    REASONING = "reasoning"


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: str = "default"


class QueryResponse(BaseModel):
    query: str
    query_type: QueryType
    answer: str
    sources: list[str] = []
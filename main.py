import logging

from fastapi import FastAPI, HTTPException

from agents.qa_agent import QAAgent
from agents.reasoning_agent import ReasoningAgent
from agents.research_agent import ResearchAgent
from agents.summary_agent import SummaryAgent
from core.orchestrator import Orchestrator
from models.schemas import QueryRequest, QueryResponse, QueryType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="AI Research Assistant")

AGENT_REGISTRY = {
    QueryType.RESEARCH: ResearchAgent(),
    QueryType.SUMMARY: SummaryAgent(),
    QueryType.QA: QAAgent(),
    QueryType.REASONING: ReasoningAgent(),
}

orchestrator = Orchestrator(agent_registry=AGENT_REGISTRY)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = orchestrator.handle_query(
            query=request.query,
            top_k=request.top_k,
            session_id=request.session_id,
        )

        return QueryResponse(
            query=request.query,
            query_type=result["query_type"],
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health():
    return {"status": "running"}
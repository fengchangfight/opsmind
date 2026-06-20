from fastapi import APIRouter, Request
from opmind.api.schemas import RetrieveRequest
from opmind.agents import RetrieveAgent

router = APIRouter(tags=["retrieve"])


@router.post("/retrieve")
async def retrieve(req: RetrieveRequest, http_request: Request):
    retrieve_agent: RetrieveAgent = http_request.app.state.runtime["retrieve"]
    results, citations, latency = await retrieve_agent.retrieve(
        query=req.query,
        top_k=req.top_k,
        filters=req.filters,
    )
    return {
        "query": req.query,
        "results": [
            {
                "chunk_id": r.chunk_id,
                "content": r.content,
                "doc_title": r.doc_title,
                "score": round(r.score, 4),
            }
            for r in results
        ],
        "citations": [c.model_dump() for c in citations],
        "latency_ms": round(latency * 1000, 1),
    }

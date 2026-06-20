import json
import asyncio
from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse
from typing import Optional
from opmind.agents import RetrieveAgent, ReasonAgent

router = APIRouter(tags=["query"])


@router.get("/query")
async def query(
    http_request: Request,
    q: str = Query(..., min_length=1, max_length=4096, alias="query"),
    top_k: int = Query(default=5, ge=1, le=20),
    category: Optional[str] = Query(default=None),
):
    runtime = http_request.app.state.runtime
    retrieve_agent: RetrieveAgent = runtime["retrieve"]
    reason_agent: ReasonAgent = runtime["reason"]

    filters = None
    if category:
        filters = {"category": category}

    results, citations, retrieve_latency = await retrieve_agent.retrieve(
        query=q,
        top_k=top_k,
        filters=filters,
    )

    async def event_generator():
        try:
            yield f"event: agent_start\ndata: {json.dumps({'agent_id': 'retrieve'})}\n\n"

            retrieval_data = {
                "num_results": len(results),
                "latency_ms": round(retrieve_latency * 1000, 1),
            }
            yield f"event: retrieval_result\ndata: {json.dumps(retrieval_data)}\n\n"

            yield f"event: agent_start\ndata: {json.dumps({'agent_id': 'reason'})}\n\n"

            full_answer = ""
            async for token in reason_agent.reason_stream(q, results, citations):
                full_answer += token
                yield f"event: chunk\ndata: {json.dumps({'content': token})}\n\n"
                await asyncio.sleep(0)

            final_data = {
                "answer": full_answer,
                "citations": [c.model_dump() for c in citations],
                "num_sources": len(results),
                "model": reason_agent.model,
            }
            yield f"event: final_answer\ndata: {json.dumps(final_data, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'code': 'INTERNAL', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

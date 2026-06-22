import json
import asyncio
import base64
from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse
from typing import Optional
from app.agents import RetrieveAgent, ReasonAgent
from app.persistence import get_repo

router = APIRouter(tags=["query"])


@router.get("/query")
async def query(
    http_request: Request,
    q: str = Query(..., min_length=1, max_length=4096, alias="query"),
    top_k: int = Query(default=5, ge=1, le=20),
    category: Optional[str] = Query(default=None),
    history: Optional[str] = Query(default=None, description="Base64 JSON chat history"),
    session_id: Optional[str] = Query(default=None),
):
    runtime = http_request.app.state.runtime
    retrieve_agent: RetrieveAgent = runtime["retrieve"]
    reason_agent: ReasonAgent = runtime["reason"]
    repo = get_repo()

    # Ensure session exists
    sid = session_id or repo.create_session()

    # Save user message
    repo.save_message(sid, "user", q)
    if not session_id:
        repo.auto_title(sid, q)

    filters = None
    if category:
        filters = {"category": category}

    # Load history from DB (preferred) or from client (fallback)
    messages_history = repo.get_messages_for_llm(sid)
    if history and len(messages_history) <= 2:
        try:
            messages_history = json.loads(base64.b64decode(history).decode())
        except Exception:
            pass

    results, citations, retrieve_latency = await retrieve_agent.retrieve(
        query=q,
        top_k=top_k,
        filters=filters,
    )

    async def event_generator():
        try:
            yield f"event: agent_start\ndata: {json.dumps({'agent_id': 'retrieve', 'session_id': sid})}\n\n"

            retrieval_data = {
                "num_results": len(results),
                "latency_ms": round(retrieve_latency * 1000, 1),
            }
            yield f"event: retrieval_result\ndata: {json.dumps(retrieval_data)}\n\n"

            yield f"event: agent_start\ndata: {json.dumps({'agent_id': 'reason'})}\n\n"

            full_answer = ""
            async for token in reason_agent.reason_stream(q, results, citations, messages_history):
                full_answer += token
                yield f"event: chunk\ndata: {json.dumps({'content': token})}\n\n"
                await asyncio.sleep(0)

            # Save assistant response
            repo.save_message(sid, "assistant", full_answer, [c.model_dump() for c in citations])

            final_data = {
                "answer": full_answer,
                "citations": [c.model_dump() for c in citations],
                "num_sources": len(results),
                "model": reason_agent.model,
                "session_id": sid,
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

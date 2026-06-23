import json
import asyncio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from app.api.schemas import ResumeRequest
from app.agents import RetrieveAgent, ReasonAgent

router = APIRouter(tags=["resume"])

_in_memory_sessions: dict[str, dict] = {}


@router.post("/resume")
async def resume(req: ResumeRequest, http_request: Request):
    session = _in_memory_sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    runtime = http_request.app.state.runtime
    reason_agent: ReasonAgent = runtime["reason"]

    results = session["results"]
    citations = session["citations"]
    modified_query = f"{session['query']}\n\nAdditional context from user: {req.human_input}"

    async def event_generator():
        try:
            yield f"event: agent_start\ndata: {json.dumps({'agent_id': 'reason', 'status': 'resumed'})}\n\n"
            full_answer = ""
            async for token in reason_agent.reason_graph_stream(modified_query, results, citations):
                full_answer += token
                yield f"event: chunk\ndata: {json.dumps({'content': token})}\n\n"
                await asyncio.sleep(0)

            yield f"event: final_answer\ndata: {json.dumps({'answer': full_answer, 'resumed': True}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'code': 'INTERNAL', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )

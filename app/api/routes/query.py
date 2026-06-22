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
    history: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
):
    runtime = http_request.app.state.runtime
    retrieve_agent: RetrieveAgent = runtime["retrieve"]
    reason_agent: ReasonAgent = runtime["reason"]
    repo = get_repo()

    user_id = getattr(http_request.state, "user_id", "default")
    sid = session_id or repo.create_session(user_id)
    repo.save_message(sid, "user", q)
    if not session_id:
        repo.auto_title(sid, q)

    filters = None
    if category:
        filters = {"category": category}

    messages_history = repo.get_messages_for_llm_with_compaction(sid)
    if history and len(messages_history) <= 2:
        try:
            messages_history = json.loads(base64.b64decode(history).decode())
        except Exception:
            pass

    results, citations, retrieve_latency = await retrieve_agent.retrieve(query=q, top_k=top_k, filters=filters)

    # Queue for tool events (cross-asyncio-task communication)
    tool_event_queue: asyncio.Queue = asyncio.Queue()

    # Start reasoning in background
    async def run_reason():
        full_answer = ""
        async for token in reason_agent.reason_graph_stream(
            q, results, citations, messages_history, tool_event_queue,
            retriever=lambda q, k: retrieve_agent.retrieve(q, k),
            session_id=sid,
        ):
            full_answer += token
        tool_event_queue.put_nowait(("done", {"answer": full_answer}))

    reason_task = asyncio.create_task(run_reason())

    async def event_generator():
        try:
            yield f"event: agent_start\ndata: {json.dumps({'agent_id': 'retrieve', 'session_id': sid})}\n\n"
            yield f"event: retrieval_result\ndata: {json.dumps({'num_results': len(results), 'latency_ms': round(retrieve_latency * 1000, 1)})}\n\n"
            yield f"event: agent_start\ndata: {json.dumps({'agent_id': 'reason'})}\n\n"

            full_answer = ""
            while True:
                try:
                    event_type, data = await asyncio.wait_for(tool_event_queue.get(), timeout=120)
                except asyncio.TimeoutError:
                    yield f"event: error\ndata: {json.dumps({'code': 'TIMEOUT', 'message': '请求超时'})}\n\n"
                    break

                if event_type == "tool_call_start":
                    yield f"event: tool_call\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

                elif event_type == "tool_call_result":
                    yield f"event: tool_result\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

                elif event_type == "reasoning_step":
                    yield f"event: reasoning_step\ndata: {json.dumps(data)}\n\n"

                elif event_type == "interrupted":
                    yield f"event: interrupted\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

                elif event_type == "chunk":
                    yield f"event: chunk\ndata: {json.dumps({'content': data['content']})}\n\n"
                    full_answer = data["content"]

                elif event_type == "done":
                    full_answer = data["answer"]
                    repo.save_message(sid, "assistant", full_answer, [c.model_dump() for c in citations])
                    final_data = {
                        "answer": full_answer,
                        "citations": [c.model_dump() for c in citations],
                        "num_sources": len(results),
                        "model": reason_agent.model,
                        "session_id": sid,
                    }
                    yield f"event: final_answer\ndata: {json.dumps(final_data, ensure_ascii=False)}\n\n"

                    # Background: check if compaction is needed and persist
                    asyncio.create_task(_maybe_compact(sid, repo, reason_agent, results, citations))
                    break

                elif event_type == "error":
                    yield f"event: error\ndata: {json.dumps(data)}\n\n"
                    break

            await reason_task

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'code': 'INTERNAL', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _maybe_compact(sid: str, repo, reason_agent, results, citations):
    """
    Background task: check if session messages need compaction and persist.
    Triggered when message count exceeds threshold.
    """
    try:
        messages = repo.get_messages_for_llm(sid)
        if len(messages) < 20:
            return  # Not enough messages to warrant compaction

        total_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        if total_tokens < reason_agent.orchestrator.budget.compaction_threshold:
            return

        doc_texts = [
            f"[{c.citation_id}] {r.doc_title}\n{r.content}"
            for r, c in zip(results, citations)
        ]
        result = await reason_agent.orchestrator.compactor.compact(
            messages, reason_agent.client, reason_agent.model, reason="post_turn",
        )
        if result and result.summary:
            repo.save_compaction(
                session_id=sid,
                summary=result.summary,
                last_message_id=len(messages),
                pre_tokens=result.pre_tokens,
                post_tokens=result.post_tokens,
                reason="post_turn",
            )
    except Exception:
        pass  # Background task: never break the main flow

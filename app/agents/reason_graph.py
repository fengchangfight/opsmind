"""
ReasonGraph: LangGraph with unified tool + iterative reasoning.

Flow: evaluate → assess_confidence → (finalize | generate_gaps → re_retrieve → loop | interrupt)

evaluate_node: normal LLM call (tools allowed, no json_object restriction)
assess_confidence_node: lightweight LLM call for confidence JSON
"""
import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class ReasonState(TypedDict, total=False):
    query: str
    context: list[str]
    citations: list[dict]
    answer: str
    confidence: float
    iteration: int
    max_iterations: int
    gaps: list[str]
    status: str


def _build_evaluate_prompt(state: ReasonState) -> str:
    ctx = "\n\n---\n\n".join(state.get("context", []))
    return (
        "Answer the question based on the context below. "
        "Use tools if appropriate. Be concise.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {state['query']}"
    )


async def build_reason_graph(
    llm_client, retriever, model: str = "gpt-4o",
    checkpoint_path: str = "./data/langgraph_checkpoint.db",
    tools: list[dict] | None = None,
    tool_executor=None,  # async fn(tool_name, args) -> str
    event_queue=None,   # asyncio.Queue for SSE progress events
):
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        checkpointer = await AsyncSqliteSaver.from_conn_string(checkpoint_path)
    except Exception:
        checkpointer = MemorySaver()

    # ── Node: evaluate (normal LLM, tools allowed) ──────────────
    async def evaluate_node(state: ReasonState) -> dict:
        it = state.get("iteration", 0)
        mx = state.get("max_iterations", 1)
        if event_queue:
            event_queue.put_nowait(("reasoning_step", {
                "iteration": it, "max_iterations": mx,
                "message": "分析中...",
            }))
        messages = [
            {"role": "system", "content": (
                "You are an SRE assistant. Answer based on context. "
                "Use available tools if needed. Be concise."
            )},
            {"role": "user", "content": _build_evaluate_prompt(state)},
        ]

        # Tool call loop inside evaluate
        for tool_round in range(3):
            kwargs = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 2048}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            resp = await llm_client.chat.completions.create(**kwargs)
            choice = resp.choices[0]

            if choice.message.tool_calls and tool_executor:
                for tc in choice.message.tool_calls:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    if event_queue:
                        event_queue.put_nowait(("tool_call_start", {
                            "tool_name": tc.function.name,
                            "arguments": args,
                        }))
                    try:
                        result = await tool_executor(tc.function.name, args)
                    except Exception as tool_err:
                        result = f"Tool error: {tool_err}"
                    if event_queue:
                        event_queue.put_nowait(("tool_result", {
                            "tool_name": tc.function.name,
                            "result": str(result)[:500],
                        }))
                    messages.append(choice.message.model_dump())
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                return {
                    "answer": choice.message.content or "",
                    "iteration": state.get("iteration", 0),
                    "status": "running",
                }
        return {"answer": messages[-1].get("content", ""), "iteration": state.get("iteration", 0), "status": "running"}

    # ── Node: assess_confidence ────────────────────────────────
    async def assess_confidence_node(state: ReasonState) -> dict:
        answer = state.get("answer", "")
        query = state.get("query", "")
        if not answer.strip():
            return {"confidence": 0.0, "gaps": [], "iteration": state.get("iteration", 0), "max_iterations": state.get("max_iterations", 3)}

        if event_queue:
            event_queue.put_nowait(("reasoning_step", {
                "iteration": state.get("iteration", 0),
                "max_iterations": state.get("max_iterations", 1),
                "message": "评估置信度...",
            }))

        prompt = (
            f'Question: {query}\nAnswer: {answer[:2000]}\n\n'
            'Rate confidence 0-1 based on evidence coverage and source consistency. '
            'Return JSON: {"confidence": 0.0-1.0, "gaps": ["gap1", ...]}'
        )
        resp = await llm_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0.1, max_tokens=256,
        )
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            data = {}
        return {
                "confidence": float(data.get("confidence", 0.5)),
                "gaps": data.get("gaps", []),
                "iteration": state.get("iteration", 0),
                "max_iterations": state.get("max_iterations", 1),
        }

    # ── Route ──────────────────────────────────────────────────
    def route_confidence(state: ReasonState) -> str:
        c = state.get("confidence", 0.0)
        i = state.get("iteration", 0)
        mx = state.get("max_iterations", 0)
        if c >= 0.7 or i >= mx:
            return "finalize"
        return "generate_gaps"

    # ── Nodes: gaps / re_retrieve / finalize / interrupt ────────
    async def generate_gaps_node(state: ReasonState) -> dict:
        return {"iteration": state.get("iteration", 0) + 1}

    async def re_retrieve_node(state: ReasonState) -> dict:
        gaps = state.get("gaps", []) or [state.get("query", "")]
        q = f"{state.get('query','')} {' '.join(gaps)}"
        if event_queue:
            event_queue.put_nowait(("reasoning_step", {
                "iteration": state.get("iteration", 0),
                "max_iterations": state.get("max_iterations", 1),
                "message": f"重新检索: {', '.join(gaps[:3])}",
            }))
        results, citations, _ = await retriever(q, 3)
        new_ctx = state.get("context", []) + [
            f"[{c.citation_id if hasattr(c,'citation_id') else '?'}] {r.doc_title if hasattr(r,'doc_title') else str(r)}"
            for r, c in zip(results, citations if citations else [])
        ]
        return {"context": new_ctx}

    async def finalize_node(state: ReasonState) -> dict:
        return {"status": "finalized"}

    # ── Build ─────────────────────────────────────────────────
    g = StateGraph(ReasonState)
    g.add_node("evaluate", evaluate_node)
    g.add_node("assess_confidence", assess_confidence_node)
    g.add_node("generate_gaps", generate_gaps_node)
    g.add_node("re_retrieve", re_retrieve_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("evaluate")
    g.add_edge("evaluate", "assess_confidence")
    g.add_conditional_edges("assess_confidence", route_confidence, {
        "finalize": "finalize", "generate_gaps": "generate_gaps",
    })
    g.add_edge("generate_gaps", "re_retrieve")
    g.add_edge("re_retrieve", "evaluate")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)

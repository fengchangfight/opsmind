"""
ReasonGraph: LangGraph-based iterative reasoning with confidence assessment.
Natural fit — the PRD already defines "置信度 < 0.7 → 重新检索" as a loop + conditional.

Nodes: evaluate → assess_confidence → generate_gaps (loop) → finalize
Edges: conditional on confidence score + iteration count.
Interrupt: when max iterations reached with low confidence, pause for human input.
"""
from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt

REASON_GRAPH_NAME = "opsmind_reason"


class ReasonState(TypedDict, total=False):
    query: str
    context: list[str]        # Retrieved document snippets
    citations: list[dict]
    answer: str               # Intermediate or final answer
    confidence: float         # 0-1 confidence score
    iteration: int            # Current iteration (0-based)
    max_iterations: int       # Default 3
    gaps: list[str]           # Knowledge gaps to fill
    status: str               # "running" | "interrupted" | "finalized"
    human_feedback: str       # User feedback on interrupt


def _build_evaluate_prompt(state: ReasonState) -> str:
    ctx = "\n\n---\n\n".join(state.get("context", []))
    return (
        "Based on the following context, answer the question concisely "
        "and assess your confidence (0-1) based on evidence coverage and source consistency.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {state['query']}"
    )


async def build_reason_graph(
    llm_client,           # AsyncOpenAI
    retriever,            # async fn(query: str, top_k: int) -> (results, citations)
    model: str = "gpt-4o",
    checkpoint_path: str = "./data/langgraph_checkpoint.db",
) -> StateGraph:
    """Build and compile the ReasonGraph (async)."""
    import json

    # Try async SQLite, fallback to in-memory
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        checkpointer = await AsyncSqliteSaver.from_conn_string(checkpoint_path)
    except Exception:
        checkpointer = MemorySaver()

    # ── Node: evaluate ─────────────────────────────────────
    async def evaluate_node(state: ReasonState) -> dict:
        """Call LLM to generate answer + confidence."""
        prompt = _build_evaluate_prompt(state)
        response = await llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "You are an SRE assistant. Answer the question based on context. "
                    "Return JSON: {\"answer\": \"...\", \"confidence\": 0.0-1.0, \"gaps\": [\"...\"]}"
                )},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=1024,
        )
        try:
            data = json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            data = {"answer": "Failed to parse LLM response", "confidence": 0.0, "gaps": []}

        return {
            "answer": data.get("answer", ""),
            "confidence": float(data.get("confidence", 0.5)),
            "gaps": data.get("gaps", []),
            "iteration": state.get("iteration", 0),
            "status": "running",
        }

    # ── Node: assess_confidence ────────────────────────────
    async def assess_confidence_node(state: ReasonState) -> dict:
        """Decide next step based on confidence + iteration count."""
        confidence = state.get("confidence", 0.0)
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 3)

        return {
            "confidence": confidence,
            "iteration": iteration,
            "max_iterations": max_iter,
        }

    # ── Route: confidence gate ─────────────────────────────
    def route_confidence(state: ReasonState) -> str:
        confidence = state.get("confidence", 0.0)
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 3)

        if confidence >= 0.7:
            return "finalize"
        elif iteration < max_iter:
            return "generate_gaps"
        else:
            return "interrupt"

    # ── Node: generate_gaps ────────────────────────────────
    async def generate_gaps_node(state: ReasonState) -> dict:
        """Augment query with gap-related keywords for re-retrieval."""
        return {
            "iteration": state.get("iteration", 0) + 1,
        }

    # ── Node: re_retrieve ────────────────────────────────
    async def re_retrieve_node(state: ReasonState) -> dict:
        """Re-query vector store with expanded keywords from gaps."""
        gaps = state.get("gaps", [])
        if not gaps:
            gaps = [state.get("query", "")]
        expanded_query = state.get("query", "") + " " + " ".join(gaps)
        results, citations = await retriever(expanded_query, 3)
        new_context = [r.content if hasattr(r, 'content') else str(r) for r in results]
        return {
            "context": state.get("context", []) + new_context,
            "citations": state.get("citations", []) + [c.model_dump() if hasattr(c, 'model_dump') else c for c in citations],
        }

    # ── Node: finalize ────────────────────────────────────
    async def finalize_node(state: ReasonState) -> dict:
        """Mark as complete."""
        return {"status": "finalized"}

    # ── Node: interrupt ────────────────────────────────────
    async def interrupt_node(state: ReasonState) -> dict:
        """Pause execution, wait for human feedback."""
        user_input = interrupt({
            "reason": f"置信度不足 ({state.get('confidence', 0):.2f})，需要人工确认",
            "confidence": state.get("confidence", 0),
            "gaps": state.get("gaps", []),
        })
        return {
            "human_feedback": str(user_input) if user_input else "",
            "iteration": state.get("iteration", 0) + 1,
        }

    # ── Build Graph ────────────────────────────────────────
    graph = StateGraph(ReasonState)

    graph.add_node("evaluate", evaluate_node)
    graph.add_node("assess_confidence", assess_confidence_node)
    graph.add_node("generate_gaps", generate_gaps_node)
    graph.add_node("re_retrieve", re_retrieve_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("interrupt", interrupt_node)

    graph.set_entry_point("evaluate")
    graph.add_edge("evaluate", "assess_confidence")

    graph.add_conditional_edges(
        "assess_confidence",
        route_confidence,
        {
            "finalize": "finalize",
            "generate_gaps": "generate_gaps",
            "interrupt": "interrupt",
        },
    )

    graph.add_edge("generate_gaps", "re_retrieve")
    graph.add_edge("re_retrieve", "evaluate")     # Loop
    graph.add_edge("interrupt", "evaluate")        # Resume after human input

    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)

from typing import Optional
from openai import AsyncOpenAI
from app.models import SearchResult, Citation
from app.config import settings
from app.context import ContextOrchestrator
from app.mcp import McpManager
from app.tools import ToolRegistry


SYSTEM_PROMPT = """You are OpsMind, an expert SRE / DevOps assistant. Answer the user's question based on the provided context documents.

Rules:
- Answer in Chinese if the user asks in Chinese, otherwise in English.
- Base your answer ONLY on the provided context. If the context is insufficient, say so.
- Cite sources using the format [1], [2], etc. matching the citation numbers.
- Be concise but thorough. Include specific steps, commands, or configurations when relevant.
- If there are multiple possible interpretations, mention them.
- Format your answer with Markdown for readability (headers, lists, code blocks)."""


class ReasonAgent:
    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self, orchestrator: ContextOrchestrator | None = None, mcp_manager: McpManager | None = None, tool_registry: ToolRegistry | None = None):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.llm_model
        self.orchestrator = orchestrator or ContextOrchestrator()
        self.mcp_manager = mcp_manager
        self.tool_registry = tool_registry
        self._reason_graph = None  # Lazily built

    async def reason_graph_stream(
        self,
        query: str,
        results: list[SearchResult],
        citations: list[Citation],
        history: list[dict] | None = None,
        event_queue=None,
        retriever=None,
        session_id: str = "",
    ):
        """
        Unified LangGraph reasoning: tools + iterative deep-dive.

        evaluate (tool-loop inside) → assess_confidence
            → finalize | generate_gaps → re_retrieve → loop | interrupt
        """
        from app.agents.reason_graph import build_reason_graph

        if retriever is None:
            retriever = lambda q, k: (results, citations)

        # Collect tools
        all_tools: list[dict] = []
        if self.mcp_manager:
            all_tools.extend(self.mcp_manager.get_all_tools())
        if self.tool_registry:
            all_tools.extend(self.tool_registry.get_all_openai_functions())

        async def tool_executor(tool_name: str, args: dict) -> str:
            if self.tool_registry and self.tool_registry.get(tool_name):
                return await self.tool_registry.execute(tool_name, args)
            if self.mcp_manager:
                return await self.mcp_manager.call_tool(tool_name, args)
            return f"Tool '{tool_name}' not found"

        if self._reason_graph is None:
            self._reason_graph = await build_reason_graph(
                llm_client=self.client,
                retriever=retriever,
                model=self.model,
                checkpoint_path=f"./data/langgraph_{session_id or 'default'}.db",
                tools=all_tools if all_tools else None,
                tool_executor=tool_executor if all_tools else None,
                event_queue=event_queue,
            )

        doc_texts = [
            f"[{c.citation_id}] {r.doc_title}\n{r.content}"
            for r, c in zip(results, citations)
        ]

        config = {"configurable": {"thread_id": session_id or "default"}}

        initial_state: dict = {
            "query": query,
            "context": doc_texts,
            "citations": [c.model_dump() for c in citations],
            "answer": "",
            "confidence": 0.0,
            "iteration": 0,
            "max_iterations": 3,
            "gaps": [],
            "status": "running",
        }

        # Check if there's an interrupted checkpoint → inject context
        try:
            last_state = await self._reason_graph.aget_state(config)
            if last_state and last_state.values:
                prev_status = last_state.values.get("status", "")
                prev_answer = last_state.values.get("answer", "")
                prev_confidence = last_state.values.get("confidence", 0)
                prev_gaps = last_state.values.get("gaps", [])
                prev_iteration = last_state.values.get("iteration", 0)

                if prev_status in ("interrupted",) or (prev_confidence < 0.7 and prev_gaps):
                    # Inject previous context into fresh state
                    initial_state["context"] = (
                        initial_state["context"] +
                        [f"[Previous iteration {prev_iteration}] Partial answer: {prev_answer}",
                         f"Knowledge gaps: {', '.join(prev_gaps)}"]
                    )
                    initial_state["iteration"] = prev_iteration + 1
                    if event_queue:
                        event_queue.put_nowait(("reasoning_step", {
                            "step": prev_iteration,
                            "confidence": prev_confidence,
                            "status": "resuming",
                            "message": "从上一轮迭代恢复，注入知识缺口...",
                        }))
        except Exception:
            pass  # No checkpoint yet, fresh start

        if event_queue:
            event_queue.put_nowait(("agent_start", {"agent_id": "reason_graph"}))
            event_queue.put_nowait(("reasoning_step", {
                "step": 0, "confidence": 0.0, "iteration": 0, "max_iterations": 3,
                "message": "开始迭代推理...",
            }))

        try:
            final_state = await self._reason_graph.ainvoke(initial_state, config)
        except Exception as e:
            # Interrupt — the graph paused at interrupt node
            if "interrupt" in str(e).lower():
                if event_queue:
                    event_queue.put_nowait(("interrupted", {
                        "reason": f"置信度不足，需要人工确认",
                        "confidence": 0.0,
                        "options": ["continue", "modify", "transfer"],
                    }))
                return
            raise

        answer = final_state.get("answer", "")
        confidence = final_state.get("confidence", 0)
        status = final_state.get("status", "finalized")

        if event_queue:
            event_queue.put_nowait(("reasoning_step", {
                "step": final_state.get("iteration", 0),
                "confidence": confidence,
                "status": status,
            }))

        yield answer

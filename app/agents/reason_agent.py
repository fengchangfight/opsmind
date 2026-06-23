import json
import time
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
        Unified LangGraph path — tools + iterative reasoning in one graph.

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

    async def reason_stream(
        self,
        query: str,
        results: list[SearchResult],
        citations: list[Citation],
        history: list[dict] | None = None,
        event_queue=None,  # asyncio.Queue for SSE tool events
    ):
        """
        Stream LLM response with context + MCP tool loop.
        
        Flow:
        1. Build base messages with context
        2. Call LLM with MCP tools → if tool call, execute via McpManager, feed result back
        3. Loop until LLM produces final text (no more tool calls)
        4. Stream final answer token by token
        """
        doc_texts = [
            f"[{c.citation_id}] Source: {r.doc_title}\nContent: {r.content}"
            for r, c in zip(results, citations)
        ]

        base_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            base_messages.extend(history)

        messages = await self.orchestrator.build_context(
            messages=base_messages,
            retrieved_docs=doc_texts,
            llm_client=self.client,
            model=self.model,
        )
        messages.append({"role": "user", "content": query})

        # MCP + native tools
        mcp_tools = self.mcp_manager.get_all_tools() if self.mcp_manager else []
        native_tools = self.tool_registry.get_all_openai_functions() if self.tool_registry else []
        all_tools = mcp_tools + native_tools

        # Tool call loop (max 3 rounds)
        for _ in range(3):
            call_kwargs: dict = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2048,
            }
            if all_tools:
                call_kwargs["tools"] = all_tools
                call_kwargs["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**call_kwargs)
            choice = response.choices[0]

            # Check for tool calls
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_name = tc.function.name
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except Exception:
                        arguments = {}

                    # Emit tool event
                    if event_queue:
                        event_queue.put_nowait(("tool_call_start", {
                            "tool_name": tool_name,
                            "arguments": arguments,
                        }))

                    # Execute: native first, then MCP
                    tool_result: str
                    if self.tool_registry and self.tool_registry.get(tool_name):
                        tool_result = await self.tool_registry.execute(tool_name, arguments)
                    elif self.mcp_manager:
                        tool_result = await self.mcp_manager.call_tool(tool_name, arguments)
                    else:
                        tool_result = f"Error: Tool '{tool_name}' not found in any registry"

                    # Emit result
                    if event_queue:
                        event_queue.put_nowait(("tool_call_result", {
                            "tool_name": tool_name,
                            "result": tool_result[:1000],
                        }))

                    # Append tool call + result to messages
                    messages.append(choice.message.model_dump())
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })
            else:
                # No more tool calls → stream final answer
                final_answer = choice.message.content or ""

                # Stream the answer as chunk events
                if event_queue:
                    event_queue.put_nowait(("chunk", {"content": final_answer}))
                yield final_answer
                return

        # If loop exhausted (too many tool calls), yield last response
        last_msg = messages[-1]
        if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
            yield last_msg.get("content", "")
        else:
            yield "Tool execution completed. Please review the results above."

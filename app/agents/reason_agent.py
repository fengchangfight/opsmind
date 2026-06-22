import json
import time
from typing import Optional
from openai import AsyncOpenAI
from app.models import SearchResult, Citation
from app.config import settings
from app.context import ContextOrchestrator
from app.mcp import McpManager


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

    def __init__(self, orchestrator: ContextOrchestrator | None = None, mcp_manager: McpManager | None = None):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.llm_model
        self.orchestrator = orchestrator or ContextOrchestrator()
        self.mcp_manager = mcp_manager

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

        # MCP tools available?
        mcp_tools = self.mcp_manager.get_all_tools() if self.mcp_manager else []

        # Tool call loop (max 3 rounds)
        for _ in range(3):
            call_kwargs: dict = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 2048,
            }
            if mcp_tools:
                call_kwargs["tools"] = mcp_tools
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

                    # Execute via MCP
                    tool_result = await self.mcp_manager.call_tool(tool_name, arguments)

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

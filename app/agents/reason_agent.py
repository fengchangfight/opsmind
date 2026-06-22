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
    ):
        """Stream LLM response with context-aware message building."""
        # Build document texts for retrieval injection
        doc_texts = [
            f"[{c.citation_id}] Source: {r.doc_title}\nContent: {r.content}"
            for r, c in zip(results, citations)
        ]

        # Build base messages: system + history
        base_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            base_messages.extend(history)

        # Use orchestrator to handle compaction + doc injection
        messages = await self.orchestrator.build_context(
            messages=base_messages,
            retrieved_docs=doc_texts,
            llm_client=self.client,
            model=self.model,
        )

        # Append the actual user query as the final message
        messages.append({"role": "user", "content": query})

        # Build call params
        call_kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 2048,
            "stream": True,
        }

        # Include MCP tools if manager is connected with tools
        if self.mcp_manager:
            mcp_tools = self.mcp_manager.get_all_tools()
            if mcp_tools:
                call_kwargs["tools"] = mcp_tools
                call_kwargs["tool_choice"] = "auto"

        stream = await self.client.chat.completions.create(**call_kwargs)

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

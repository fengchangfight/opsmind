import json
import time
from openai import AsyncOpenAI
from opmind.models import SearchResult, Citation
from opmind.config import settings


SYSTEM_PROMPT = """You are OpsMind, an expert SRE / DevOps assistant. Answer the user's question based on the provided context documents.

Rules:
- Answer in Chinese if the user asks in Chinese, otherwise in English.
- Base your answer ONLY on the provided context. If the context is insufficient, say so.
- Cite sources using the format [1], [2], etc. matching the citation numbers.
- Be concise but thorough. Include specific steps, commands, or configurations when relevant.
- If there are multiple possible interpretations, mention them.
- Format your answer with Markdown for readability (headers, lists, code blocks)."""


class ReasonAgent:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.llm_model

    async def reason(
        self,
        query: str,
        results: list[SearchResult],
        citations: list[Citation],
    ) -> dict:
        t0 = time.monotonic()

        context = self._build_context(results, citations)
        user_message = self._build_user_message(query, context)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=2048,
        )

        answer = response.choices[0].message.content or ""
        latency = time.monotonic() - t0

        return {
            "answer": answer,
            "citations": [c.model_dump() for c in citations],
            "latency_seconds": round(latency, 2),
            "model": self.model,
            "num_sources": len(results),
        }

    async def reason_stream(
        self,
        query: str,
        results: list[SearchResult],
        citations: list[Citation],
    ):
        context = self._build_context(results, citations)
        user_message = self._build_user_message(query, context)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=2048,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _build_context(self, results: list[SearchResult], citations: list[Citation]) -> str:
        parts = []
        for i, (r, c) in enumerate(zip(results, citations)):
            parts.append(
                f"[{c.citation_id}] Source: {r.doc_title}\n"
                f"Content: {r.content}\n"
                f"Relevance: {r.score:.2f}"
            )
        return "\n\n---\n\n".join(parts)

    def _build_user_message(self, query: str, context: str) -> str:
        return f"""Context documents:
{context}

User question: {query}

Please answer based on the context above. Cite sources with [number] notation."""

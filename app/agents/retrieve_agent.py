import time
import json
from typing import Optional
from app.models import SearchResult, Citation
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.reranker import Reranker
from app.config import settings


QUERY_EXPANSION_PROMPT = """Expand the following search query into 3-5 alternative versions 
to improve retrieval recall. Vary keywords and phrasing while preserving intent.
Return JSON: {"variants": ["variant1", "variant2", ...]}

Original query: {query}"""


class RetrieveAgent:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = Reranker()
        self._llm_client = None

    def set_llm_client(self, client):
        """Set LLM client for query expansion."""
        self._llm_client = client

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> tuple[list[SearchResult], list[Citation], float]:
        t0 = time.monotonic()
        top_k = top_k or settings.top_k

        # P0: Query Expansion (LLM generates 3-5 variants for better recall)
        expanded_queries = [query]
        if self._llm_client:
            try:
                resp = await self._llm_client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(query=query)}],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=256,
                )
                data = json.loads(resp.choices[0].message.content or "{}")
                variants = data.get("variants", [])
                if variants:
                    expanded_queries.extend(variants[:4])
            except Exception:
                pass  # Query expansion is best-effort

        # Embed + search each variant, collect unique results
        all_results: list[SearchResult] = []
        seen = set()
        for q in expanded_queries:
            query_embedding = await self.embedder.embed_single(q)
            results = self.vector_store.search(query_embedding, top_k=top_k * 2, filters=filters)
            for r in results:
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    all_results.append(r)

        # P0: Cross-Encoder Reranker (Top-N → fine scores → Top-K)
        if len(all_results) > top_k:
            all_results = self.reranker.rerank_results(query, all_results, top_n=top_k * 3)
            all_results = all_results[:top_k]

        citations = []
        for i, r in enumerate(all_results[:top_k]):
            citations.append(Citation(
                citation_id=str(i + 1),
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                doc_title=r.doc_title,
                excerpt=r.content[:300],
                relevance_score=r.rerank_score if hasattr(r, 'rerank_score') else r.score,
            ))

        latency = time.monotonic() - t0
        return all_results[:top_k], citations, latency

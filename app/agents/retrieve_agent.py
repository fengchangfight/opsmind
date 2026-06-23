"""RetrieveAgent — LlamaIndex hybrid retriever (dense + sparse via RRF) + Reranker + Query Expansion."""
import json, time
from typing import Optional
from app.models import SearchResult, Citation
from app.retrieval.reranker import Reranker
from app.config import settings


QUERY_EXPANSION_PROMPT = """Generate 3-5 search query variants for the question below. 
Return JSON: {"variants": ["variant1", "variant2", ...]}
Question: {query}"""


class RetrieveAgent:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = Reranker()
        self._llm_client = None
        self._li_retriever = None

    def set_llm_client(self, client):
        self._llm_client = client

    def init_li_retriever(self):
        """Build LlamaIndex hybrid retriever at startup."""
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        Settings.embed_model = FastEmbedEmbedding(model_name=settings.embedding_dense_model)

        li_store = self.vector_store.get_li_store()
        index = VectorStoreIndex.from_vector_store(li_store)
        self._li_retriever = index.as_retriever(
            vector_store_query_mode="hybrid",
            similarity_top_k=settings.top_k * 3,
        )
        return self

    async def retrieve(
        self, query: str, top_k: int | None = None, filters: dict | None = None,
    ) -> tuple[list[SearchResult], list[Citation], float]:
        t0 = time.monotonic()
        top_k = top_k or settings.top_k

        # Query Expansion (LLM generates 3-5 variants)
        expanded = [query]
        if self._llm_client:
            try:
                resp = await self._llm_client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(query=query)}],
                    response_format={"type": "json_object"}, temperature=0.3, max_tokens=256,
                )
                data = json.loads(resp.choices[0].message.content or "{}")
                expanded.extend(data.get("variants", [])[:4])
            except Exception:
                pass

        # LlamaIndex hybrid retriever (dense + sparse via Milvus RRF)
        all_results = []
        seen = set()
        for q in expanded:
            nodes = self._li_retriever.retrieve(q)
            for node in nodes:
                nid = node.node.node_id
                if nid not in seen:
                    seen.add(nid)
                    all_results.append(SearchResult(
                        chunk_id=nid if nid else node.node.metadata.get("chunk_id", nid),
                        doc_id=node.node.metadata.get("doc_id", ""),
                        content=node.node.get_content() or node.node.text,
                        doc_title=node.node.metadata.get("doc_title", ""),
                        score=node.score or 0.0,
                        metadata={"category": node.node.metadata.get("category", "")},
                    ))
        # Cross-Encoder Reranker
        if len(all_results) > top_k:
            all_results = self.reranker.rerank_results(query, all_results, top_n=top_k * 3)
            all_results = all_results[:top_k]

        citations = [
            Citation(citation_id=str(i+1), chunk_id=r.chunk_id, doc_id=r.doc_id,
                     doc_title=r.doc_title, excerpt=r.content[:300],
                     relevance_score=r.rerank_score if hasattr(r, 'rerank_score') else r.score)
            for i, r in enumerate(all_results[:top_k])
        ]
        return all_results[:top_k], citations, time.monotonic() - t0

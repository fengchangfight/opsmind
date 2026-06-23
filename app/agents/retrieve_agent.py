"""RetrieveAgent — LlamaIndex VectorStoreIndex retriever + Query Expansion + Reranker."""
import time, json
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
        self._li_retriever = None  # Pre-built by init_li_retriever()

    def set_llm_client(self, client):
        self._llm_client = client

    def init_li_retriever(self):
        """Build LlamaIndex retriever at startup (before event loop)."""
        from llama_index.core import VectorStoreIndex
        from llama_index.embeddings.fastembed import FastEmbedEmbedding

        li_store = self.vector_store.get_li_store()
        embed = FastEmbedEmbedding(model_name=settings.embedding_dense_model)
        index = VectorStoreIndex.from_vector_store(li_store, embed_model=embed)
        self._li_retriever = index.as_retriever(similarity_top_k=settings.top_k * 3)
        return self

    def _get_li_retriever(self):
        """Lazy-init LlamaIndex retriever with hybrid search."""
        if self._li_retriever is None:
            li_store = self.vector_store.get_li_store()
            embed_model = FastEmbedEmbedding(model_name=settings.embedding_dense_model)
            index = VectorStoreIndex.from_vector_store(li_store, embed_model=embed_model)
            self._li_retriever = index.as_retriever(similarity_top_k=settings.top_k * 3)
        return self._li_retriever

    async def retrieve(
        self, query: str, top_k: int | None = None, filters: dict | None = None,
    ) -> tuple[list[SearchResult], list[Citation], float]:
        t0 = time.monotonic()
        top_k = top_k or settings.top_k

        # Query Expansion (async LLM call)
        expanded_queries = [query]
        if self._llm_client:
            try:
                resp = await self._llm_client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(query=query)}],
                    response_format={"type": "json_object"}, temperature=0.3, max_tokens=256,
                )
                data = json.loads(resp.choices[0].message.content or "{}")
                expanded_queries.extend(data.get("variants", [])[:4])
            except Exception:
                pass

        # LlamaIndex dense + Sparse hybrid search
        all_results: list[SearchResult] = []
        seen = set()

        for q in expanded_queries:
            # LlamaIndex dense search via pre-built retriever
            if self._li_retriever:
                nodes = self._li_retriever.retrieve(q)
                for node in nodes:
                    nid = node.node.node_id
                    if nid not in seen:
                        seen.add(nid)
                        all_results.append(SearchResult(
                            chunk_id=nid, doc_id=node.node.metadata.get("doc_id", ""),
                            content=node.node.text, doc_title=node.node.metadata.get("doc_title", ""),
                            score=node.score or 0.0,
                            metadata={"category": node.node.metadata.get("category", "")},
                        ))
            # Sparse hybrid search for keyword recall
            dense = await self.embedder.embed_single(q)
            sparse = await self.embedder.embed_sparse_single(q)
            results = self.vector_store.hybrid_search(dense, sparse, top_k=top_k * 3, filters=filters)
            for r in results:
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    all_results.append(r)

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

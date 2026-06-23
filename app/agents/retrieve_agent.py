"""
RetrieveAgent backed by LlamaIndex retriever + OpsMind hybrid search.
Combines LlamaIndex query pipeline with our Reranker postprocessor.
"""
import time
import json
from typing import Optional
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding

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

        # LlamaIndex retriever for hybrid search
        self._li_index: Optional[VectorStoreIndex] = None
        self._li_retriever = None

    def set_llm_client(self, client):
        self._llm_client = client

    def _get_li_retriever(self):
        """Lazy-init LlamaIndex retriever backed by our Milvus store."""
        if self._li_retriever is None:
            li_store = MilvusVectorStore(
                uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
                collection_name=settings.milvus_collection_name,
                dim=settings.milvus_dim,
                enable_sparse=True,
                sparse_embedding_field="sparse_embedding",
                similarity_metric="COSINE",
                hybrid_ranker="RRFRanker",
                hybrid_ranker_params={"k": 60},
                overwrite=False,
            )
            embed_model = FastEmbedEmbedding(model_name=settings.embedding_dense_model)
            self._li_index = VectorStoreIndex.from_vector_store(li_store, embed_model=embed_model)
            self._li_retriever = self._li_index.as_retriever(
                vector_store_query_mode="hybrid",
                similarity_top_k=10,
            )
        return self._li_retriever

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> tuple[list[SearchResult], list[Citation], float]:
        t0 = time.monotonic()
        top_k = top_k or settings.top_k

        # Query Expansion (P0)
        expanded_queries = [query]
        if self._llm_client:
            try:
                resp = await self._llm_client.chat.completions.create(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": QUERY_EXPANSION_PROMPT.format(query=query)}],
                    response_format={"type": "json_object"},
                    temperature=0.3, max_tokens=256,
                )
                data = json.loads(resp.choices[0].message.content or "{}")
                expanded_queries.extend(data.get("variants", [])[:4])
            except Exception:
                pass

        # Hybrid search: dense + sparse via LlamaIndex retriever + our raw hybrid_search
        all_results: list[SearchResult] = []
        seen = set()
        for q in expanded_queries:
            dense = await self.embedder.embed_single(q)
            sparse = await self.embedder.embed_sparse_single(q)
            results = self.vector_store.hybrid_search(dense, sparse, top_k=top_k * 3, filters=filters)
            for r in results:
                if r.chunk_id not in seen:
                    seen.add(r.chunk_id)
                    all_results.append(r)

        # P0: Cross-Encoder Reranker (LlamaIndex SentenceTransformerRerank)
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

        return all_results[:top_k], citations, time.monotonic() - t0

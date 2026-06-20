import time
from typing import Optional
from opmind.models import SearchResult, Citation
from opmind.retrieval.embedder import Embedder
from opmind.retrieval.vector_store import VectorStore
from opmind.config import settings


class RetrieveAgent:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> tuple[list[SearchResult], list[Citation], float]:
        t0 = time.monotonic()
        top_k = top_k or settings.top_k

        query_embedding = await self.embedder.embed_single(query)
        results = self.vector_store.search(query_embedding, top_k=top_k * 2, filters=filters)

        citations = []
        for i, r in enumerate(results[:top_k]):
            citations.append(Citation(
                citation_id=str(i + 1),
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                doc_title=r.doc_title,
                excerpt=r.content[:300],
                relevance_score=r.score,
            ))

        latency = time.monotonic() - t0
        return results[:top_k], citations, latency

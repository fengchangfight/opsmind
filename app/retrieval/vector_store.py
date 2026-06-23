"""
Milvus vector store backed by LlamaIndex MilvusVectorStore.
Supports dense + sparse hybrid search via RRFRanker.
"""
from typing import Optional
from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker, DataType
from llama_index.vector_stores.milvus import MilvusVectorStore
from app.config import settings
from app.models import Chunk, SearchResult


COLLECTION = settings.milvus_collection_name
DENSE_FIELD = "embedding"
SPARSE_FIELD = "sparse_embedding"
DIM = settings.milvus_dim


class VectorStore:
    """Vector store backed by LlamaIndex MilvusVectorStore for collection management,
       with custom hybrid search for dense + sparse raw vector queries."""

    def __init__(self):
        self._store = self._build_store()
        self._pymilvus = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")

    def _build_store(self) -> MilvusVectorStore:
        return MilvusVectorStore(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
            collection_name=COLLECTION,
            dim=DIM,
            embedding_field=DENSE_FIELD,
            enable_sparse=False,  # Sparse handled separately via our Embedder
            similarity_metric="COSINE",
            overwrite=False,
            output_fields=["chunk_id", "doc_id", "content", "doc_title", "category"],
        )

    # ── Collection ──────────────────────────────────────────

    def _ensure_collection(self):
        # LlamaIndex auto-creates on first access
        pass

    def count(self) -> int:
        try:
            stats = self._pymilvus.get_collection_stats(COLLECTION)
            return stats.get("row_count", 0)
        except Exception:
            return 0

    def clear(self):
        try:
            if self._pymilvus.has_collection(COLLECTION):
                self._pymilvus.drop_collection(COLLECTION)
        except Exception:
            pass

    # ── Write ───────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]):
        """Insert chunks with dense + sparse embeddings."""
        if not chunks:
            return
        data = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "content": c.content[:8192],
                DENSE_FIELD: c.embedding or [0.0] * DIM,
                SPARSE_FIELD: c.sparse_embedding or {},
                "doc_title": c.metadata.get("doc_title", ""),
                "category": c.metadata.get("category", ""),
            }
            for c in chunks
        ]
        self._pymilvus.insert(COLLECTION, data)
        self._pymilvus.flush(COLLECTION)

    def delete_by_doc_id(self, doc_id: str):
        self._pymilvus.delete(COLLECTION, f'doc_id == "{doc_id}"')

    # ── Search ──────────────────────────────────────────────

    def search(
        self, query_embedding: list[float], top_k: int = 5, filters: dict | None = None,
    ) -> list[SearchResult]:
        """Dense-only vector search."""
        expr = self._build_expr(filters) if filters else None
        results = self._pymilvus.search(
            collection_name=COLLECTION, data=[query_embedding],
            anns_field=DENSE_FIELD,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k, filter=expr,
            output_fields=["chunk_id", "doc_id", "content", "doc_title", "category"],
        )
        return self._parse_results(results)

    def hybrid_search(
        self, dense_vector: list[float], sparse_vector: dict[int, float],
        top_k: int = 5, filters: dict | None = None,
    ) -> list[SearchResult]:
        """Dense + sparse hybrid search with RRFRanker."""
        expr = self._build_expr(filters) if filters else None
        dense_req = AnnSearchRequest(
            data=[dense_vector], anns_field=DENSE_FIELD,
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k * 3, expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vector], anns_field=SPARSE_FIELD,
            param={"metric_type": "IP"}, limit=top_k * 3, expr=expr,
        )
        results = self._pymilvus.hybrid_search(
            collection_name=COLLECTION, reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60), limit=top_k,
            output_fields=["chunk_id", "doc_id", "content", "doc_title", "category"],
        )
        return self._parse_results(results)

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _build_expr(filters: dict) -> str:
        return " && ".join(f'{k} == "{v}"' for k, v in filters.items())

    @staticmethod
    def _parse_results(results) -> list[SearchResult]:
        parsed = []
        for batch in results:
            for hit in batch:
                entity = hit.get("entity", hit)
                parsed.append(SearchResult(
                    chunk_id=str(entity.get("chunk_id", "")),
                    doc_id=str(entity.get("doc_id", "")),
                    content=str(entity.get("content", "")),
                    doc_title=str(entity.get("doc_title", "")),
                    score=float(hit.get("distance", hit.get("score", 0))),
                    metadata={"category": str(entity.get("category", ""))},
                ))
        return parsed

"""
Milvus vector store with hybrid search (dense + sparse).
Uses MilvusClient API — dense HNSW index + sparse inverted index + RRFRanker fusion.
"""
from typing import Optional
from pymilvus import (
    MilvusClient, DataType, AnnSearchRequest, RRFRanker,
)
from app.config import settings
from app.models import Chunk, SearchResult


COLLECTION = settings.milvus_collection_name
DENSE_FIELD = "embedding_dense"
SPARSE_FIELD = "embedding_sparse"
DIM = settings.milvus_dim


class VectorStore:
    def __init__(self):
        self._client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")

    def _ensure_collection(self):
        if self._client.has_collection(COLLECTION):
            return

        schema = self._client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=8192)
        schema.add_field(field_name=DENSE_FIELD, datatype=DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field(field_name=SPARSE_FIELD, datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="doc_title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=64)

        idx = self._client.prepare_index_params()
        idx.add_index(field_name=DENSE_FIELD, metric_type="COSINE", index_type="HNSW",
                      params={"M": 16, "efConstruction": 200})
        idx.add_index(field_name=SPARSE_FIELD, metric_type="IP", index_type="SPARSE_INVERTED_INDEX")

        self._client.create_collection(COLLECTION, schema=schema, index_params=idx)
        self._client.load_collection(COLLECTION)

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        self._ensure_collection()
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
        self._client.insert(COLLECTION, data)
        self._client.flush(COLLECTION)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Dense-only vector search."""
        self._ensure_collection()
        expr = self._build_expr(filters) if filters else None
        results = self._client.search(
            collection_name=COLLECTION,
            data=[query_embedding],
            anns_field=DENSE_FIELD,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            filter=expr,
            output_fields=["chunk_id", "doc_id", "content", "doc_title", "category"],
        )
        return self._parse_results(results)

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """
        Hybrid search: dense + sparse with RRFRanker fusion.
        Sends two AnnSearchRequests → Milvus RRFRanker(k=60) → merged Top-K.
        """
        self._ensure_collection()
        expr = self._build_expr(filters) if filters else None

        dense_req = AnnSearchRequest(
            data=[dense_vector],
            anns_field=DENSE_FIELD,
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k * 3,
            expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field=SPARSE_FIELD,
            param={"metric_type": "IP"},
            limit=top_k * 3,
            expr=expr,
        )

        results = self._client.hybrid_search(
            collection_name=COLLECTION,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=top_k,
            output_fields=["chunk_id", "doc_id", "content", "doc_title", "category"],
        )
        return self._parse_results(results)

    def count(self) -> int:
        try:
            stats = self._client.get_collection_stats(COLLECTION)
            return stats.get("row_count", 0)
        except Exception:
            return 0

    def clear(self):
        try:
            if self._client.has_collection(COLLECTION):
                self._client.drop_collection(COLLECTION)
        except Exception:
            pass

    @staticmethod
    def _build_expr(filters: dict) -> str:
        conds = [f'{k} == "{v}"' for k, v in filters.items()]
        return " && ".join(conds)

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

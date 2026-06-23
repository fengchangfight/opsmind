"""
Milvus vector store — dense (HNSW) + sparse (BM25) hybrid search via RRFRanker.
Uses pymilvus directly for collection management; LlamaIndex MilvusVectorStore
available via _get_li_store() for advanced retriever/query engine use.
"""
from typing import Optional
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker
from app.config import settings
from app.models import Chunk, SearchResult


COLLECTION = settings.milvus_collection_name
DENSE_FIELD = "embedding"
SPARSE_FIELD = "sparse_embedding"
DIM = settings.milvus_dim


class VectorStore:
    def __init__(self):
        self._client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")

    def _ensure_collection(self):
        if self._client.has_collection(COLLECTION):
            return
        schema = self._client.create_schema(auto_id=True)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=256)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field("content", DataType.VARCHAR, max_length=8192)
        schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("doc_title", DataType.VARCHAR, max_length=512)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        idx = self._client.prepare_index_params()
        idx.add_index(DENSE_FIELD, metric_type="COSINE", index_type="HNSW", params={"M": 16, "efConstruction": 200})
        idx.add_index(SPARSE_FIELD, metric_type="IP", index_type="SPARSE_INVERTED_INDEX")
        self._client.create_collection(COLLECTION, schema=schema, index_params=idx)
        self._client.load_collection(COLLECTION)

    def count(self) -> int:
        try:
            return self._client.get_collection_stats(COLLECTION).get("row_count", 0)
        except Exception:
            return 0

    def clear(self):
        try:
            if self._client.has_collection(COLLECTION):
                self._client.drop_collection(COLLECTION)
        except Exception:
            pass

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        self._ensure_collection()
        data = [{
            "chunk_id": c.chunk_id, "doc_id": c.doc_id, "content": c.content[:8192],
            DENSE_FIELD: c.embedding or [0.0] * DIM,
            SPARSE_FIELD: c.sparse_embedding or {},
            "doc_title": c.metadata.get("doc_title", ""),
            "category": c.metadata.get("category", ""),
        } for c in chunks]
        self._client.insert(COLLECTION, data)
        self._client.flush(COLLECTION)

    def delete_by_doc_id(self, doc_id: str):
        self._ensure_collection()
        self._client.delete(COLLECTION, f'doc_id == "{doc_id}"')

    def search(self, query_embedding: list[float], top_k: int = 5, filters: dict | None = None) -> list[SearchResult]:
        self._ensure_collection()
        expr = self._expr(filters)
        results = self._client.search(
            COLLECTION, [query_embedding], anns_field=DENSE_FIELD,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k, filter=expr,
            output_fields=["chunk_id", "doc_id", "content", "doc_title", "category"],
        )
        return self._parse(results)

    def hybrid_search(self, dense: list[float], sparse: dict[int, float], top_k: int = 5, filters: dict | None = None) -> list[SearchResult]:
        self._ensure_collection()
        expr = self._expr(filters)
        dense_req = AnnSearchRequest([dense], DENSE_FIELD, {"metric_type": "COSINE", "params": {"ef": 64}}, top_k * 3, expr)
        sparse_req = AnnSearchRequest([sparse], SPARSE_FIELD, {"metric_type": "IP"}, top_k * 3, expr)
        results = self._client.hybrid_search(
            COLLECTION, [dense_req, sparse_req], RRFRanker(k=60), top_k,
            output_fields=["chunk_id", "doc_id", "content", "doc_title", "category"],
        )
        return self._parse(results)

    # ── LlamaIndex integration point ───────────────────────
    def get_li_store(self):
        """Return a LlamaIndex MilvusVectorStore backed by this collection (for VectorStoreIndex, retriever, etc.)."""
        from llama_index.vector_stores.milvus import MilvusVectorStore
        return MilvusVectorStore(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
            collection_name=COLLECTION, dim=DIM, enable_sparse=False,
            similarity_metric="COSINE", overwrite=False,
        )

    # ── Helpers ────────────────────────────────────────────
    @staticmethod
    def _expr(filters: dict | None) -> str | None:
        return " && ".join(f'{k} == "{v}"' for k, v in filters.items()) if filters else None

    @staticmethod
    def _parse(results) -> list[SearchResult]:
        parsed = []
        for batch in results:
            for hit in batch:
                e = hit.get("entity", hit)
                parsed.append(SearchResult(
                    chunk_id=str(e.get("chunk_id", "")), doc_id=str(e.get("doc_id", "")),
                    content=str(e.get("content", "")), doc_title=str(e.get("doc_title", "")),
                    score=float(hit.get("distance", hit.get("score", 0))),
                    metadata={"category": str(e.get("category", ""))},
                ))
        return parsed

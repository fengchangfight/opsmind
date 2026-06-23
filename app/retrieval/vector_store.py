"""
Milvus vector store with custom schema — dense + sparse hybrid search.
LlamaIndex MilvusVectorStore for retrieval only (via get_li_store).
Schema self-managed because LlamaIndex doesn't handle our custom scalar fields.
"""
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
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        # doc_title, category stored as dynamic fields
        idx = self._client.prepare_index_params()
        idx.add_index(DENSE_FIELD, metric_type="COSINE", index_type="HNSW", params={"M": 16, "efConstruction": 200})
        idx.add_index(SPARSE_FIELD, metric_type="IP", index_type="SPARSE_INVERTED_INDEX")
        self._client.create_collection(COLLECTION, schema=schema, index_params=idx)
        self._client.load_collection(COLLECTION)

    def count(self) -> int:
        try:
            stats = self._client.get_collection_stats(COLLECTION)
            return stats.get("row_count", 0) or 0
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
            "id": c.chunk_id, "doc_id": c.doc_id, "text": c.content[:8192],
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

    # ── LlamaIndex integration (retrieval only) ─────────────
    def get_li_store(self):
        """LlamaIndex MilvusVectorStore for retrieval only (dense, no sparse — our collection has the data)."""
        from llama_index.vector_stores.milvus import MilvusVectorStore
        return MilvusVectorStore(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
            collection_name=COLLECTION, dim=DIM,
            similarity_metric="COSINE", overwrite=False,
            text_key="text",
            output_fields=["id", "doc_id", "text", "doc_title", "category"],
        )

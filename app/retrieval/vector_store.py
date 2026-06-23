"""Milvus vector store — LlamaIndex MilvusVectorStore with scalar fields for doc_title/category."""
from pymilvus import MilvusClient, DataType
from app.config import settings
from app.models import Chunk


COLLECTION = settings.milvus_collection_name
DIM = settings.milvus_dim


class VectorStore:
    def __init__(self):
        self._client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")

    def count(self) -> int:
        try:
            # get_collection_stats is unreliable; query a single row to check data exists
            self._client.load_collection(COLLECTION)
            res = self._client.query(COLLECTION, filter="id != ''", output_fields=["id"], limit=1)
            return 1 if res else 0  # approximation — real count needs iteration
        except Exception:
            return 0

    def clear(self):
        try:
            if self._client.has_collection(COLLECTION):
                self._client.drop_collection(COLLECTION)
        except Exception:
            pass

    def add_chunks(self, chunks: list[Chunk]):
        """Write path: pymilvus direct insert (dense + sparse + scalar fields)."""
        if not chunks:
            return
        self._ensure_collection()
        data = [{
            "id": c.chunk_id, "doc_id": c.doc_id, "text": c.content[:8192],
            "embedding": c.embedding or [0.0] * DIM,
            "sparse_embedding": c.sparse_embedding or {},
            "doc_title": c.metadata.get("doc_title", ""),
            "category": c.metadata.get("category", ""),
        } for c in chunks]
        self._client.insert(COLLECTION, data)
        self._client.flush(COLLECTION)

    def delete_by_doc_id(self, doc_id: str):
        self._ensure_collection()
        self._client.delete(COLLECTION, f'doc_id == "{doc_id}"')

    # ── LlamaIndex MilvusVectorStore ────────────────────────

    def get_li_store(self):
        """LlamaIndex MilvusVectorStore with scalar fields + sparse support."""
        from llama_index.vector_stores.milvus import MilvusVectorStore
        from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction

        class DummySparse(BaseSparseEmbeddingFunction):
            def encode_queries(self, qs): return [{}] * len(qs)
            def encode_documents(self, ds): return self.encode_queries(ds)

        return MilvusVectorStore(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
            collection_name=COLLECTION, dim=DIM,
            similarity_metric="COSINE", overwrite=False,
            text_key="text",
            enable_sparse=True,
            sparse_embedding_function=DummySparse(),
            scalar_field_names=["doc_title", "category"],
            scalar_field_types=[DataType.VARCHAR, DataType.VARCHAR],
            output_fields=["id", "doc_id", "text", "doc_title", "category"],
        )

    def _ensure_collection(self):
        if self._client.has_collection(COLLECTION):
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=256)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("doc_title", DataType.VARCHAR, max_length=512)
        schema.add_field("category", DataType.VARCHAR, max_length=64)
        idx = self._client.prepare_index_params()
        idx.add_index("embedding", metric_type="COSINE", index_type="HNSW", params={"M": 16, "efConstruction": 200})
        idx.add_index("sparse_embedding", metric_type="IP", index_type="SPARSE_INVERTED_INDEX")
        self._client.create_collection(COLLECTION, schema=schema, index_params=idx)
        self._client.load_collection(COLLECTION)

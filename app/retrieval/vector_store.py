"""Milvus vector store — schema managed directly (sparse field needs manual index), reads/writes via LlamaIndex."""
from pymilvus import MilvusClient, DataType
from app.config import settings


COLLECTION = settings.milvus_collection_name
DIM = settings.milvus_dim


class VectorStore:
    def __init__(self):
        self._client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
        self._li = None  # LlamaIndex MilvusVectorStore, lazy

    # ── Schema management (requires direct pymilvus for SPARSE_FLOAT_VECTOR + indices) ──

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

    def clear(self):
        self._ensure_collection()  # triggers the check, won't recreate if exists
        if self._client.has_collection(COLLECTION):
            self._client.drop_collection(COLLECTION)
        self._li = None  # invalidate LlamaIndex store

    def delete_by_doc_id(self, doc_id: str):
        self._ensure_collection()
        self._client.delete(COLLECTION, f'doc_id == "{doc_id}"')

    # ── LlamaIndex MilvusVectorStore (read + write via IngestionPipeline) ──

    def get_li_store(self):
        if self._li is not None:
            return self._li
        from llama_index.vector_stores.milvus import MilvusVectorStore
        from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction

        class DummySparse(BaseSparseEmbeddingFunction):
            def encode_queries(self, qs): return [{}] * len(qs)
            def encode_documents(self, ds): return self.encode_queries(ds)

        self._li = MilvusVectorStore(
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
        return self._li

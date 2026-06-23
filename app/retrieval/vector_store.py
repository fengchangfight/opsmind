"""
Milvus vector store — schema managed by LlamaIndex MilvusVectorStore.
Dense + sparse hybrid search via RRFRanker, BM25 sparse via custom func.
"""
from pymilvus import MilvusClient
from app.config import settings
from app.models import Chunk


COLLECTION = settings.milvus_collection_name
DENSE_FIELD = "embedding"
SPARSE_FIELD = "sparse_embedding"
DIM = settings.milvus_dim


class VectorStore:
    def __init__(self):
        self._client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
        self._li_store = None

    @property
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

    # ── Write (compatible with LlamaIndex schema) ──────────

    def add_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return
        # LlamaIndex schema: id(VARCHAR, pk), doc_id, text, embedding, sparse_embedding
        data = [{
            "id": c.chunk_id,
            "doc_id": c.doc_id,
            "text": c.content[:8192],
            DENSE_FIELD: c.embedding or [0.0] * DIM,
            SPARSE_FIELD: c.sparse_embedding or {},
            "doc_title": c.metadata.get("doc_title", ""),
            "category": c.metadata.get("category", ""),
        } for c in chunks]
        self._ensure_li_store()
        self._client.insert(COLLECTION, data)
        self._client.flush(COLLECTION)

    def delete_by_doc_id(self, doc_id: str):
        self._ensure_li_store()
        self._client.delete(COLLECTION, f'doc_id == "{doc_id}"')

    # ── LlamaIndex MilvusVectorStore ────────────────────────

    def _ensure_li_store(self):
        """Create LlamaIndex store. Schema auto-created on first access."""
        self.get_li_store()

    def get_li_store(self):
        """LlamaIndex MilvusVectorStore with BM25 sparse (no FlagEmbedding)."""
        if self._li_store is not None:
            return self._li_store

        from llama_index.vector_stores.milvus import MilvusVectorStore
        from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction

        class BM25SparseFunc(BaseSparseEmbeddingFunction):
            def __init__(self):
                from fastembed import SparseTextEmbedding
                self._model = SparseTextEmbedding(model_name=settings.embedding_sparse_model)

            def encode_queries(self, queries):
                results = []
                for emb in self._model.embed(list(queries)):
                    results.append(dict(zip(emb.indices.tolist(), emb.values.tolist())))
                return results

            def encode_documents(self, docs):
                return self.encode_queries(docs)

        self._li_store = MilvusVectorStore(
            uri=f"http://{settings.milvus_host}:{settings.milvus_port}",
            collection_name=COLLECTION, dim=DIM,
            enable_sparse=True,
            sparse_embedding_function=BM25SparseFunc(),
            similarity_metric="COSINE",
            hybrid_ranker="RRFRanker",
            hybrid_ranker_params={"k": 60},
            overwrite=False,
            text_key="text",
            doc_id_field="doc_id",
            output_fields=["id", "doc_id", "text", "doc_title", "category"],
        )
        return self._li_store

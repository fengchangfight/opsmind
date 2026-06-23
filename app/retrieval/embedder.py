"""Embedding wrapper backed by LlamaIndex FastEmbedEmbedding for dense, fastembed directly for sparse."""
from typing import Optional
from fastembed import SparseTextEmbedding


_dense: Optional["FastEmbedEmbedding"] = None
_sparse: Optional[SparseTextEmbedding] = None


def _get_dense():
    global _dense
    if _dense is None:
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        from app.config import settings
        _dense = FastEmbedEmbedding(model_name=settings.embedding_dense_model)
    return _dense


def _get_sparse():
    global _sparse
    if _sparse is None:
        from app.config import settings
        _sparse = SparseTextEmbedding(model_name=settings.embedding_sparse_model)
    return _sparse


class Embedder:
    """Dense + sparse embedding generator using LlamaIndex + fastembed."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return _get_dense().get_text_embedding_batch(texts)

    async def embed_single(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]

    async def embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        if not texts:
            return []
        model = _get_sparse()
        results = []
        for emb in model.embed(texts):
            indices = emb.indices.tolist() if hasattr(emb, 'indices') else list(emb.keys())
            values = emb.values.tolist() if hasattr(emb, 'values') else list(emb.values())
            results.append(dict(zip(indices, values)))
        return results

    async def embed_sparse_single(self, text: str) -> dict[int, float]:
        results = await self.embed_sparse([text])
        return results[0] if results else {}

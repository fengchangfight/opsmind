from typing import Optional
from fastembed import TextEmbedding
from app.config import settings

# Global lazy-loaded embedder
_embedder: Optional[TextEmbedding] = None


def _get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=settings.embedding_model)
    return _embedder


class Embedder:
    def __init__(self):
        pass

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = _get_embedder()
        embeddings = list(model.embed(texts))
        return [emb.tolist() for emb in embeddings]

    async def embed_single(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]

"""Cross-Encoder Reranker using sentence_transformers directly (fast, cached)."""
import os
import logging

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_OFFLINE", "1")


class Reranker:
    """Cross-Encoder re-ranker using sentence_transformers CrossEncoder (fast, local)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", top_n: int = 5):
        self.model_name = model_name
        self.top_n = top_n
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"[Reranker] Loaded {self.model_name}")
        except Exception as e:
            logger.warning(f"[Reranker] unavailable: {e}")
            self._model = False

    def rerank(self, query: str, documents: list[str], top_n: int = 5) -> list[tuple[str, float]]:
        """Re-rank by cross-encoding (query, doc) pairs."""
        self._load()
        if self._model is False or not documents:
            return [(d, 0.5) for d in documents[:top_n]]
        scores = self._model.predict([(query, d) for d in documents], show_progress_bar=False)
        return sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)[:top_n]

    def rerank_results(self, query: str, results: list, top_n: int = 5) -> list:
        """Re-rank SearchResult objects. Returns re-ordered list with updated scores."""
        if not results:
            return results
        docs = [r.content for r in results]
        ranked = self.rerank(query, docs, top_n)
        score_map = {doc: score for doc, score in ranked}
        for r in results:
            r.rerank_score = score_map.get(r.content, r.score)
            r.score = r.rerank_score
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_n]

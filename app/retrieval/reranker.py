"""
Cross-Encoder Reranker using BGE-Reranker (ONNX or transformers).
Top-50 coarse results → BGE-Reranker fine scores → Top-5.
LLD-04 §3.3.3 — P0 requirement.
"""
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-Encoder re-ranker.
    Uses sentence-transformers CrossEncoder (BGE-Reranker-Large or MiniLM).
    Falls back gracefully if model not available.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"[Reranker] Loaded {self.model_name}")
        except Exception as e:
            logger.warning(f"[Reranker] CrossEncoder unavailable: {e}")
            self._model = False

    def rerank(self, query: str, documents: list[str], top_n: int = 5) -> list[tuple[str, float]]:
        """
        Re-rank documents by cross-encoding (query, doc) pairs.
        
        Args:
            query: user query string
            documents: list of document texts
            top_n: number of top results to return
        
        Returns:
            list of (document, score) sorted by relevance descending
        """
        self._load_model()
        if self._model is False or not documents:
            return [(d, 0.5) for d in documents[:top_n]]

        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)
        return ranked[:top_n]

    def rerank_results(self, query: str, results: list, top_n: int = 5) -> list:
        """
        Re-rank SearchResult objects. Returns re-ordered list with updated scores.
        """
        if not results:
            return results

        docs = [r.content for r in results]
        ranked = self.rerank(query, docs, top_n)

        score_map = {doc: score for doc, score in ranked}
        for r in results:
            r.rerank_score = score_map.get(r.content, r.score)
            r.score = r.rerank_score

        return sorted(results, key=lambda r: r.score, reverse=True)[:top_n]

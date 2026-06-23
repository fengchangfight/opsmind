"""Cross-Encoder Reranker — background warming, non-blocking at query time."""
import logging
import threading

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", top_n: int = 5):
        self.model_name = model_name
        self.top_n = top_n
        self._postprocessor = None
        self._warming = False

    def warm_up(self):
        """Start background model loading. Non-blocking."""
        if self._warming or self._postprocessor is not None:
            return
        self._warming = True

        def _load():
            try:
                from llama_index.core.postprocessor import SentenceTransformerRerank
                self._postprocessor = SentenceTransformerRerank(model=self.model_name, top_n=self.top_n)
                logger.info(f"[Reranker] Ready: {self.model_name}")
            except Exception as e:
                logger.warning(f"[Reranker] unavailable: {e}")
                self._postprocessor = False

        threading.Thread(target=_load, daemon=True).start()

    def rerank_results(self, query: str, results: list, top_n: int = 5) -> list:
        """Re-rank results. If model not ready, return top_n as-is (non-blocking)."""
        if not results or self._postprocessor is None:
            return results[:top_n]
        if self._postprocessor is False:
            return results[:top_n]

        from llama_index.core.schema import NodeWithScore, TextNode
        nodes = [NodeWithScore(node=TextNode(text=r.content), score=r.score) for r in results]
        ranked = self._postprocessor.postprocess_nodes(nodes, query_str=query)
        score_map = {n.node.text: n.score for n in ranked}
        for r in results:
            r.rerank_score = score_map.get(r.content, r.score)
            r.score = r.rerank_score
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_n]

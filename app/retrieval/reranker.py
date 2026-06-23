"""
Cross-Encoder Reranker backed by LlamaIndex SentenceTransformerRerank.
LLD-04 §3.3.3 — P0 requirement.
"""
import os
import logging

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_OFFLINE", "1")


class Reranker:
    """Cross-Encoder re-ranker using LlamaIndex SentenceTransformerRerank backend."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", top_n: int = 5):
        self.model_name = model_name
        self.top_n = top_n
        self._postprocessor = None

    def _load(self):
        if self._postprocessor is not None:
            return
        try:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            self._postprocessor = SentenceTransformerRerank(
                model=self.model_name,
                top_n=self.top_n,
            )
            logger.info(f"[Reranker] Loaded {self.model_name} via LlamaIndex")
        except Exception as e:
            logger.warning(f"[Reranker] unavailable: {e}")
            self._postprocessor = False

    def rerank(self, query: str, documents: list[str], top_n: int = 5) -> list[tuple[str, float]]:
        """Re-rank by cross-encoding (query, doc) pairs."""
        self._load()
        if self._postprocessor is False or not documents:
            return [(d, 0.5) for d in documents[:top_n]]

        # LlamaIndex SentenceTransformerRerank works on nodes
        from llama_index.core.schema import NodeWithScore, TextNode

        nodes = [NodeWithScore(node=TextNode(text=doc), score=0.5) for doc in documents]
        ranked = self._postprocessor.postprocess_nodes(nodes, query_str=query)
        return [(n.node.text, n.score) for n in ranked[:top_n]]

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

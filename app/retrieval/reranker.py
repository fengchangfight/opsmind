"""Reranker — currently disabled for demo stability. Set ENABLE_RERANKER=true to activate."""
import logging

logger = logging.getLogger(__name__)
ENABLED = False  # Disabled: model load causes timeouts on slow networks


class Reranker:
    def __init__(self, model_name: str = "", top_n: int = 5):
        self.top_n = top_n

    def warm_up(self):
        pass  # Disabled

    def rerank_results(self, query: str, results: list, top_n: int = 5) -> list:
        return results[:top_n]

"""
Token budget allocator for LLM context window.
Splits available tokens across: system prompt, retrieval docs, history, output buffer.
"""
from dataclasses import dataclass


@dataclass
class TokenBudget:
    context_window: int = 128_000       # Model context window size
    output_buffer: int = 20_000         # Reserved for LLM output
    safety_margin: int = 5_000          # Safety buffer
    system_tokens: int = 5_000          # System prompt + memory (fixed)

    # Allocation ratios (of usable space)
    history_ratio: float = 0.45          # ~45% to conversation history
    retrieval_ratio: float = 0.30        # ~30% to retrieved documents
    tail_ratio: float = 0.55             # ~55% of history kept verbatim (tail)

    @property
    def usable(self) -> int:
        """Tokens available for history + retrieval after fixed overhead."""
        return max(0, self.context_window - self.output_buffer - self.safety_margin)

    @property
    def history_budget(self) -> int:
        return max(1000, int(self.usable * self.history_ratio))

    @property
    def retrieval_budget(self) -> int:
        return max(500, int(self.usable * self.retrieval_ratio))

    @property
    def tail_budget(self) -> int:
        """Tokens to keep verbatim at conversation tail."""
        return max(500, int(self.history_budget * self.tail_ratio))

    @property
    def summary_budget(self) -> int:
        """Max tokens for compaction summary (~history - tail)."""
        return max(200, self.history_budget - self.tail_budget)

    @property
    def compaction_threshold(self) -> int:
        """Trigger compaction when (system + history) exceeds 70% of usable."""
        return int(self.usable * 0.70)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count: 4 chars per token (OpenCode convention)."""
        return max(1, len(text) // 4)

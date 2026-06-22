"""
Compaction trigger: multi-path (preflight, post-turn, overflow, manual).
Wires TokenBudget + ConversationCompactor into the request lifecycle.
"""
from app.context.token_budget import TokenBudget
from app.context.conversation_compactor import ConversationCompactor, CompactionResult


class CompactionTrigger:
    def __init__(
        self,
        budget: TokenBudget | None = None,
        compactor: ConversationCompactor | None = None,
    ):
        self.budget = budget or TokenBudget()
        self.compactor = compactor or ConversationCompactor(self.budget)
        self._consecutive_overflows: int = 0

    async def preflight(
        self, messages: list[dict], llm_client, model: str,
    ) -> tuple[list[dict], CompactionResult | None]:
        """
        Check before API call. If token budget exceeded, compact proactively.
        Returns: (possibly_compacted_messages, compaction_result_or_none)
        """
        total_tokens = sum(
            self.budget.estimate_tokens(str(m.get("content", "")))
            for m in messages
        )
        if not self.compactor.should_compact(total_tokens):
            return messages, None

        result = await self.compactor.compact(
            messages, llm_client, model, reason="preflight",
        )
        if result.summary:
            return self.compactor.build_compact_messages(messages, result), result
        return messages, None

    async def overflow(
        self, messages: list[dict], llm_client, model: str,
    ) -> tuple[list[dict], CompactionResult | None]:
        """
        Aggressive compaction on context-overflow error.
        Reduces tail_budget for emergency space.
        """
        self._consecutive_overflows += 1
        self.budget.tail_budget = max(200, self.budget.tail_budget // 2)
        self.budget.output_buffer += 10_000  # Temporarily borrow

        result = await self.compactor.compact(
            messages, llm_client, model, reason="overflow",
        )
        if result.summary:
            return self.compactor.build_compact_messages(messages, result), result
        return messages, None

    def total_tokens(self, messages: list[dict]) -> int:
        return sum(
            self.budget.estimate_tokens(str(m.get("content", "")))
            for m in messages
        )

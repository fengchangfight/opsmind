"""
Context orchestrator: ties TokenBudget + ConversationCompactor + retrieval docs
into a single `build_context()` call used by the API layer.
"""
from app.context.token_budget import TokenBudget
from app.context.conversation_compactor import ConversationCompactor
from app.context.compaction_trigger import CompactionTrigger


class ContextOrchestrator:
    def __init__(
        self,
        budget: TokenBudget | None = None,
        compactor: ConversationCompactor | None = None,
        trigger: CompactionTrigger | None = None,
    ):
        self.budget = budget or TokenBudget()
        self.compactor = compactor or ConversationCompactor(self.budget)
        self.trigger = trigger or CompactionTrigger(self.budget, self.compactor)

    async def build_context(
        self,
        messages: list[dict],
        retrieved_docs: list[str],
        llm_client,
        model: str,
    ) -> list[dict]:
        """
        Build final LLM context messages:
        1. Preflight compaction if needed
        2. Trim retrieved docs to retrieval_budget
        3. Inject docs as system message
        """
        # Step 1: Preflight compaction
        messages, _ = await self.trigger.preflight(messages, llm_client, model)

        # Step 2: Trim retrieved docs
        doc_budget = self.budget.retrieval_budget
        docs_text = ""
        for doc in retrieved_docs:
            doc_tokens = self.budget.estimate_tokens(doc)
            if doc_budget - doc_tokens < 0:
                break
            docs_text += doc + "\n\n"
            doc_budget -= doc_tokens

        # Step 3: Inject as system message
        if docs_text:
            messages.append({
                "role": "system",
                "content": (
                    "<retrieved-documents>\n"
                    f"{docs_text}\n"
                    "</retrieved-documents>\n\n"
                    "[System note: Use ONLY the above documents to answer. "
                    "Cite sources with [number] notation.]"
                ),
            })

        return messages

    async def build_with_overflow_recovery(
        self, messages: list[dict], retrieved_docs: list[str],
        llm_client, model: str,
    ) -> list[dict]:
        """Overflow recovery: aggressive compaction then rebuild context."""
        messages, _ = await self.trigger.overflow(messages, llm_client, model)
        return await self.build_context(messages, retrieved_docs, llm_client, model)

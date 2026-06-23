"""
Conversation compactor: head/tail/summary split with LLM-powered summarization.
Inspired by: Hermes ContextCompressor, OpenCode SessionCompaction, Claude Code compact.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.context.token_budget import TokenBudget

SUMMARY_TEMPLATE = """You are summarizing an SRE/DevOps troubleshooting conversation.
Create a structured summary of the conversation history below.

<conversation-to-summarize>
{serialized}
</conversation-to-summarize>

{focus}

Generate a summary in this format:

<summary>
## Primary Topics
## Key Findings
## Decisions Made
## Files/Resources
## Unresolved Questions
## Current State
## User Messages Summary
</summary>"""


@dataclass
class CompactionResult:
    summary: str
    pre_tokens: int
    post_tokens: int
    savings_pct: float
    reason: str
    timestamp: str


class ConversationCompactor:
    def __init__(self, budget: TokenBudget | None = None):
        self.budget = budget or TokenBudget()
        self._last_summary: str = ""
        self._consecutive_ineffective: int = 0

    def should_compact(self, total_tokens: int) -> bool:
        if total_tokens < self.budget.compaction_threshold:
            return False
        if self._consecutive_ineffective >= 2:
            return False  # Circuit breaker: anti-thrashing
        return True

    def _select_head_tail(
        self,
        messages: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """
        Split messages into head (protected) and rest.
        Head = system message + first user-assistant pair.
        Tail = recent messages fitting within tail_budget (counted from end).
        Returns: (head_and_tail_kept, middle_to_summarize)
        """
        if len(messages) <= 2:
            return messages, []

        # Head: system + first exchange (protect unconditionally)
        head_end = 0
        found_first_exchange = 0
        for i, m in enumerate(messages):
            if m["role"] == "system":
                head_end = i + 1
            elif m["role"] == "user" and found_first_exchange == 0:
                found_first_exchange += 1
            elif m["role"] == "assistant" and found_first_exchange == 1:
                found_first_exchange += 1
                head_end = i + 1
            elif found_first_exchange >= 2:
                break
            else:
                head_end = i + 1

        head = messages[:head_end]
        remaining = messages[head_end:]

        if not remaining:
            return head, []

        # Tail: walk backward collecting tokens until budget exhausted
        tail_start = len(remaining)
        tail_tokens = 0
        for i in range(len(remaining) - 1, -1, -1):
            t = self.budget.estimate_tokens(str(remaining[i].get("content", "")))
            if tail_tokens + t > self.budget.tail_budget:
                tail_start = i + 1
                break
            tail_tokens += t

        # Ensure at least last user message is in tail
        last_user_idx = -1
        for i in range(len(remaining) - 1, -1, -1):
            if remaining[i]["role"] == "user":
                last_user_idx = i
                break
        if last_user_idx >= 0 and last_user_idx < tail_start:
            tail_start = last_user_idx

        middle = remaining[:tail_start]
        tail = remaining[tail_start:]

        return head + tail, middle

    async def compact(
        self,
        messages: list[dict],
        llm_client,  # openai.AsyncOpenAI
        model: str,
        focus: str = "",
        reason: str = "auto",
    ) -> CompactionResult:
        """
        Compress conversation: summarize middle, keep head + tail verbatim.
        Returns CompactionResult with summary text.
        """
        kept, middle = self._select_head_tail(messages)

        if len(middle) <= 2:
            return CompactionResult(
                summary="",
                pre_tokens=0,
                post_tokens=0,
                savings_pct=0,
                reason=reason,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        serialized = "\n\n".join(
            f"[{m['role']}]: {str(m.get('content', ''))[:1500]}" for m in middle
        )

        focus_line = f"Focus topic: {focus}" if focus else ""
        prompt = SUMMARY_TEMPLATE.format(serialized=serialized, focus=focus_line)

        # Incremental update: inject previous summary
        if self._last_summary:
            prompt += (
                f"\n\n<previous-summary>\n{self._last_summary}\n</previous-summary>\n"
                "Update this summary: preserve still-relevant details, "
                "merge new facts, remove stale information."
            )

        try:
            response = await llm_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a context summarization assistant. Respond with the summary only, no preamble.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=min(self.budget.summary_budget // 4, 2000),
                temperature=0.1,
            )
            summary = response.choices[0].message.content or ""
        except Exception:
            summary = (
                f"[Auto-summary of {len(middle)} messages omitted due to API error]"
            )

        pre_tokens = sum(
            self.budget.estimate_tokens(str(m.get("content", ""))) for m in messages
        )
        post_tokens = sum(
            self.budget.estimate_tokens(str(m.get("content", ""))) for m in kept
        ) + self.budget.estimate_tokens(summary)
        savings_pct = (pre_tokens - post_tokens) / max(pre_tokens, 1)

        if savings_pct < 0.10:
            self._consecutive_ineffective += 1
        else:
            self._consecutive_ineffective = 0

        self._last_summary = summary

        return CompactionResult(
            summary=summary,
            pre_tokens=pre_tokens,
            post_tokens=post_tokens,
            savings_pct=round(savings_pct, 3),
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def build_compact_messages(
        self,
        messages: list[dict],
        result: CompactionResult,
    ) -> list[dict]:
        """Assemble compacted message list: head + summary + tail."""
        kept, _ = self._select_head_tail(messages)

        if not result.summary:
            return kept

        # Insert summary as system message after the first system message,
        # then continue appending the rest (head + tail)
        compacted = []
        summary_inserted = False
        for m in kept:
            compacted.append(m)
            if not summary_inserted and m["role"] == "system":
                compacted.append({
                    "role": "system",
                    "content": (
                        "<conversation-summary>\n"
                        f"{result.summary}\n"
                        "</conversation-summary>\n\n"
                        "[System note: The above summarizes earlier conversation. "
                        "The following messages are the recent context.]"
                    ),
                })
                summary_inserted = True

        if not summary_inserted:
            compacted.insert(0, {
                "role": "system",
                "content": (
                    "<conversation-summary>\n"
                    f"{result.summary}\n"
                    "</conversation-summary>"
                ),
            })

        return compacted

"""Tests for ConversationCompactor — core compaction logic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.context.token_budget import TokenBudget
from app.context.conversation_compactor import ConversationCompactor, CompactionResult


def make_messages(*roles: str) -> list[dict]:
    """Create dummy messages from role sequence."""
    return [{"role": r, "content": f"Content from {r} message"} for r in roles]


def test_select_head_tail_small():
    """Messages <= 2: returns all messages, empty middle."""
    compactor = ConversationCompactor()
    msgs = make_messages("user", "assistant")
    kept, middle = compactor._select_head_tail(msgs)
    assert len(kept) == 2, f"Expected 2 kept, got {len(kept)}"
    assert len(middle) == 0, f"Expected 0 middle, got {len(middle)}"


def test_select_head_tail_splits():
    """Long conversation: head + middle + tail."""
    compactor = ConversationCompactor()
    msgs = make_messages(
        "system", "user", "assistant",  # head
        "user", "assistant", "user", "assistant", "user", "assistant",  # middle
        "user", "assistant",  # tail
    )
    kept, middle = compactor._select_head_tail(msgs)
    assert len(middle) > 0, f"Expected non-empty middle"
    assert len(kept) < len(msgs), f"Expected kept < msgs"
    assert kept[0]["role"] == "system", f"Head should start with system, got {kept[0]['role']}"
    assert kept[-1]["role"] == "assistant", f"Tail should end with assistant, got {kept[-1]['role']}"


def test_build_compact_messages_preserves_tail():
    """THE BUG — summary inserted after system, tail must be present."""
    compactor = ConversationCompactor()

    # Simulate: system + head user/assistant + some middle + tail user/assistant
    msgs = make_messages(
        "system",  # head
        "user", "assistant",  # first exchange (head)
        "user", "assistant", "user", "assistant",  # middle (would be summarized)
        "user", "assistant",  # tail
    )

    result = CompactionResult(
        summary="This is the compaction summary.",
        pre_tokens=500,
        post_tokens=200,
        savings_pct=0.6,
        reason="auto",
        timestamp="2026-01-01T00:00:00",
    )

    compacted = compactor.build_compact_messages(msgs, result)

    # The summary must exist
    summaries = [m for m in compacted if "<conversation-summary>" in m["content"]]
    assert len(summaries) == 1, f"Expected 1 summary, got {len(summaries)}"

    # System must be first
    assert compacted[0]["role"] == "system"

    # Tail messages must be present at the end
    assert compacted[-2]["role"] == "user", f"Expected tail user, got {compacted[-2]['role']}"
    assert compacted[-1]["role"] == "assistant", f"Expected tail assistant, got {compacted[-1]['role']}"

    # Summary should be after system, before tail
    summary_idx = [i for i, m in enumerate(compacted) if "<conversation-summary>" in m["content"]][0]
    tail_user_idx = len(compacted) - 2
    assert summary_idx < tail_user_idx, f"Summary ({summary_idx}) must be before tail ({tail_user_idx})"


def test_build_compact_messages_no_summary():
    """Empty summary: returns kept messages unchanged."""
    compactor = ConversationCompactor()
    msgs = make_messages("system", "user", "assistant", "user", "assistant")
    result = CompactionResult(summary="", pre_tokens=0, post_tokens=0, savings_pct=0, reason="auto", timestamp="")
    compacted = compactor.build_compact_messages(msgs, result)
    assert compacted == compactor._select_head_tail(msgs)[0]


def test_should_compact_anti_thrashing():
    """After 2 ineffective compactions, should_compact returns False."""
    compactor = ConversationCompactor()
    compactor._consecutive_ineffective = 2

    # Even if tokens exceed threshold, should not compact
    result = compactor.should_compact(999_999)
    assert result is False, f"Anti-thrashing should block compaction after 2 ineffective attempts"


def test_compact_empty_middle():
    """Middle with <=2 messages: returns empty CompactionResult."""
    import asyncio
    compactor = ConversationCompactor()
    msgs = make_messages("system", "user", "assistant", "user", "assistant")

    async def run():
        return await compactor.compact(msgs, None, "test-model")

    result = asyncio.run(run())
    assert result.summary == ""
    assert result.savings_pct == 0


if __name__ == "__main__":
    test_select_head_tail_small()
    test_select_head_tail_splits()
    test_build_compact_messages_preserves_tail()
    test_build_compact_messages_no_summary()
    test_should_compact_anti_thrashing()
    test_compact_empty_middle()
    print("All 6 compaction tests passed!")

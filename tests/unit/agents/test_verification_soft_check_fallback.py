from __future__ import annotations

import asyncio

from backend.agents.verification import SuggestionItem, _verify_inner
from backend.llm.adapter import LLMCallError


def _run(coro):
    return asyncio.run(coro)


def test_soft_check_llm_failure_keeps_deterministic_acceptance():
    items = [SuggestionItem(text="红枣性温 [source: c1]", source_ids=["c1"])]

    async def failing_complete(messages, **kwargs):
        raise LLMCallError("exhausted")

    result = _run(
        _verify_inner(
            items,
            available_source_ids=["c1"],
            branch="fact_query",
            complete=failing_complete,
        )
    )
    assert len(result.accepted) == 1
    assert result.accepted[0].text.startswith("红枣性温")
    assert result.llm_raw is None

"""Unit tests for PRD §11 naive RAG fallback (dual SubAgent failure path)."""
from __future__ import annotations

import asyncio

import pytest

from backend.agents.naive_rag_fallback import (
    NAIVE_RAG_SYSTEM_MARKER,
    run_naive_rag_fallback,
)
from backend.llm.adapter import LLMResult, ModelTier
from backend.mcp_server.tools._retrieval_common import RetrievedChunk


def _chunk(source_id: str, domain: str, text: str = "body") -> RetrievedChunk:
    return RetrievedChunk(
        source_id=source_id,
        domain=domain,
        source_file="f.jsonl",
        source_type="t",
        text=text,
        metadata={},
        score=0.8,
    )


class _FakeComplete:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[list[dict]] = []

    async def __call__(self, messages, *, tools=None, **kwargs):
        self.calls.append(messages)
        return LLMResult(
            text=self.text,
            model="fake",
            tier=ModelTier.DEV,
            provider="fake",
        )


def test_run_naive_rag_fallback_single_llm_call_and_source_ids(monkeypatch):
    def _fake_search(domain: str, query: str, **kwargs):
        if domain == "tcm":
            return [_chunk("tcm_1", "tcm", "温性食材")]
        return [_chunk("nut_1", "nutrition", "蛋白质来源")]

    monkeypatch.setattr(
        "backend.agents.naive_rag_fallback.search_knowledge_chunks",
        _fake_search,
    )
    complete = _FakeComplete("降级建议 [source: tcm_1] [source: nut_1]")
    result = asyncio.run(
        run_naive_rag_fallback(
            "今天该吃什么",
            complete,
            constitution="气虚质",
            locale="zh",
        )
    )

    assert result.domain == "naive_rag_fallback"
    assert result.tool_call_count == 0
    assert "降级建议" in result.final_text
    assert len(complete.calls) == 1
    system_prompt = complete.calls[0][0]["content"]
    assert NAIVE_RAG_SYSTEM_MARKER in system_prompt
    assert "气虚质" in system_prompt
    user_prompt = complete.calls[0][1]["content"]
    assert "tcm_1" in user_prompt
    assert "nut_1" in user_prompt

    tool_names = {m["name"] for m in result.messages if m.get("role") == "tool"}
    assert tool_names == {"retrieve_tcm", "retrieve_nutrition"}


def test_run_naive_rag_fallback_raises_on_empty_llm_text(monkeypatch):
    monkeypatch.setattr(
        "backend.agents.naive_rag_fallback.search_knowledge_chunks",
        lambda domain, query, **kwargs: [],
    )
    complete = _FakeComplete("   ")
    with pytest.raises(RuntimeError, match="empty text"):
        asyncio.run(run_naive_rag_fallback("hello", complete))

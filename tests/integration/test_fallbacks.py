"""
PRD §11 fallback rows that exist in code today.

ENGINEERING §7.1: an untested fallback does not exist. These tests cover the
actual pipeline, not the wish-list:
  - one SubAgent fails → partial_failure + the other side's text
  - both SubAgents fail → both_subagents_failed guardrail (no naive RAG)
  - 90s chain timeout → chain_timeout + done
  - empty retrieval → no static fallback table; evidence repair labels retained content
  - conflict_gaps: see tests/unit/agents/test_conflict_gaps.py (P1-7)

Mock LLM only. No live keys / network.
"""
from __future__ import annotations

import asyncio
import re

import pytest
from fastapi.testclient import TestClient

from api.main import (
    app,
    get_clarification_store,
    get_complete_fn,
    get_conflict_rules_fetcher,
    get_idle_session_folder,
    get_mcp_server,
    get_onboarding_store,
    get_pending_critical_store,
    get_session_history_loader,
    get_turn_recorder,
    get_user_profile_ensurer,
    get_user_profile_fetcher,
)
from backend.agents.clarification import InMemoryClarificationStore
from backend.agents.user_context import UserProfileContext
from backend.llm.adapter import LLMResult, ModelTier
from backend.llm.providers.base import ToolCall
from backend.mcp_server.registry import ToolDefinition, default_tool_definitions
from backend.mcp_server.server import DietExpertMcpServer
from backend.memory.pending_critical_facts import InMemoryPendingCriticalFactStore
from backend.onboarding.session_store import InMemoryOnboardingSessionStore


def _result(text="", tool_calls=None) -> LLMResult:
    return LLMResult(
        text=text, model="m", tier=ModelTier.DEV, provider="fake", tool_calls=tool_calls
    )


def _accept_soft_check() -> LLMResult:
    return _result(text='{"reject": [], "retry_reconciliation": false}')


def _chunk(source_id: str, domain: str, text: str = "x") -> dict:
    return {
        "source_id": source_id,
        "domain": domain,
        "source_file": "f",
        "source_type": "t",
        "text": text,
        "metadata": {},
        "score": 0.5,
    }


def _server_with_handlers(**handlers) -> DietExpertMcpServer:
    base = default_tool_definitions()
    tools: dict[str, ToolDefinition] = {}
    for name, tool in base.items():
        handler = handlers.get(name, lambda **kw: {"stub": True, "kwargs": kw})
        tools[name] = ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            handler=handler,
        )
    return DietExpertMcpServer(tools=tools)


def _parse_sse(body: str) -> list[tuple[str, str]]:
    events = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        m = re.search(r"event:\s*(\S+)", block)
        d = re.search(r"data:\s*(.*)", block)
        if m and d:
            events.append((m.group(1), d.group(1)))
    return events


@pytest.fixture(autouse=True)
def _clear_overrides():
    stub = UserProfileContext(user_id="default_user", onboarding_done=True)
    store = InMemoryOnboardingSessionStore()
    clarification_store = InMemoryClarificationStore()
    app.dependency_overrides[get_user_profile_fetcher] = lambda: (lambda **kw: stub)
    app.dependency_overrides[get_user_profile_ensurer] = lambda: (lambda **kw: True)
    app.dependency_overrides[get_onboarding_store] = lambda: store
    app.dependency_overrides[get_clarification_store] = lambda: clarification_store
    app.dependency_overrides[get_pending_critical_store] = lambda: InMemoryPendingCriticalFactStore()
    app.dependency_overrides[get_session_history_loader] = lambda: (lambda session_id: "")
    app.dependency_overrides[get_turn_recorder] = lambda: (lambda session_id, turn, **kw: None)
    app.dependency_overrides[get_idle_session_folder] = lambda: (lambda session_id: None)
    app.dependency_overrides[get_conflict_rules_fetcher] = lambda: (lambda *args, **kwargs: [])
    yield
    app.dependency_overrides.clear()


def _client_with(server: DietExpertMcpServer, complete) -> TestClient:
    app.dependency_overrides[get_mcp_server] = lambda: server
    app.dependency_overrides[get_complete_fn] = lambda: complete
    return TestClient(app)


class _TcmRaisesNutritionOk:
    """TCM complete() blows up; nutrition finishes with a cited conclusion."""

    async def __call__(self, messages, *, tools=None, **kwargs):
        system_text = (messages[0].get("content") or "") if messages else ""
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if "路由分类器" in system_text:
            return _result(text='{"branch":"full_recommend","domain_hint":null}')
        if "中医饮食 SubAgent" in system_text:
            raise RuntimeError("tcm provider down")
        if "营养学 SubAgent" in system_text:
            if has_tool_result:
                return _result(text="营养学侧结论 [source: n1]")
            return _result(
                tool_calls=[ToolCall(id="c1", name="retrieve_nutrition", arguments={"query": "x"})]
            )
        return _accept_soft_check()


def test_single_subagent_failure_emits_partial_failure_and_other_side():
    complete = _TcmRaisesNutritionOk()
    server = _server_with_handlers(retrieve_nutrition=lambda **kw: [_chunk("n1", "nutrition")])
    client = _client_with(server, complete)
    resp = client.post("/api/chat", json={"session_id": "fb1", "message": "今天该吃什么"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert any(e == "guardrail" and "partial_failure" in d for e, d in events)
    token_texts = "".join(d for e, d in events if e == "token")
    assert "营养学侧结论" in token_texts
    assert events[-1][0] == "done"


class _BothSidesRaise:
    async def __call__(self, messages, *, tools=None, **kwargs):
        system_text = (messages[0].get("content") or "") if messages else ""
        if "路由分类器" in system_text:
            return _result(text='{"branch":"full_recommend","domain_hint":null}')
        if "中医饮食 SubAgent" in system_text or "营养学 SubAgent" in system_text:
            raise RuntimeError("both providers down")
        return _accept_soft_check()


def test_both_subagents_fail_is_guardrail_not_naive_rag():
    """PRD wants naive RAG here. Code does not have that pipeline — test reality."""
    client = _client_with(_server_with_handlers(), _BothSidesRaise())
    resp = client.post("/api/chat", json={"session_id": "fb2", "message": "今天该吃什么"})
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    assert any(e == "guardrail" and "both_subagents_failed" in d for e, d in events)
    assert not any(e == "token" for e, _ in events)
    assert events[-1][0] == "done"


class _BothHang:
    def __init__(self) -> None:
        self.tcm_cancelled = False
        self.nutrition_cancelled = False

    async def __call__(self, messages, *, tools=None, **kwargs):
        system_text = (messages[0].get("content") or "") if messages else ""
        if "路由分类器" in system_text:
            return _result(text='{"branch":"full_recommend","domain_hint":null}')
        if "中医饮食 SubAgent" in system_text:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self.tcm_cancelled = True
                raise
        if "营养学 SubAgent" in system_text:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self.nutrition_cancelled = True
                raise
        return _accept_soft_check()


def test_chain_timeout_forced_close(monkeypatch):
    monkeypatch.setenv("CHAIN_TIMEOUT_S", "0.4")
    monkeypatch.setenv("SUBAGENT_TIMEOUT_S", "5")
    complete = _BothHang()
    client = _client_with(_server_with_handlers(), complete)
    resp = client.post("/api/chat", json={"session_id": "fb3", "message": "今天该吃什么"})
    events = _parse_sse(resp.text)
    assert any(e == "guardrail" and "chain_timeout" in d for e, d in events)
    assert events[-1][0] == "done"
    assert complete.tcm_cancelled is True
    assert complete.nutrition_cancelled is True


class _EmptyRetrievalThenUncited:
    """SubAgent retrieves nothing, then answers without a source_id."""

    async def __call__(self, messages, *, tools=None, **kwargs):
        system_text = (messages[0].get("content") or "") if messages else ""
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if "路由分类器" in system_text:
            return _result(text='{"branch":"fact_query","domain_hint":"tcm"}')
        if "中医饮食 SubAgent" in system_text:
            if has_tool_result:
                return _result(text="红枣性温，可以经常吃。")
            return _result(
                tool_calls=[ToolCall(id="c1", name="retrieve_tcm", arguments={"query": "红枣"})]
            )
        if "证据修复" in system_text:
            return _result(text="红枣的具体性质需要以可靠资料为准。")
        return _accept_soft_check()


def test_empty_retrieval_repairs_uncited_answer_with_explicit_label():
    """An uncited answer is repaired and retained with a general-knowledge label."""
    server = _server_with_handlers(retrieve_tcm=lambda **kw: [])
    client = _client_with(server, _EmptyRetrievalThenUncited())
    resp = client.post("/api/chat", json={"session_id": "fb4", "message": "红枣是什么性味"})
    events = _parse_sse(resp.text)
    token_texts = "".join(d for e, d in events if e == "token")
    assert "红枣性温" not in token_texts  # 被拒绝的原始内容不能泄漏进正文
    assert "红枣的具体性质" in token_texts
    assert "模型通用知识" in token_texts
    assert events[-1][0] == "done"

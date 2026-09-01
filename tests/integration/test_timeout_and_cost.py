"""ENGINEERING §2 pits 2–3 through `/api/chat`: subagent timeout, chain cancel, cost sum."""
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
from backend.llm.providers.base import TokenUsage, ToolCall
from backend.mcp_server.registry import ToolDefinition, default_tool_definitions
from backend.mcp_server.server import DietExpertMcpServer
from backend.observability.tracing import use_memory_backend
from backend.memory.pending_critical_facts import InMemoryPendingCriticalFactStore
from backend.onboarding.session_store import InMemoryOnboardingSessionStore


def _result(text="", tool_calls=None, usage=None, cost_est=None) -> LLMResult:
    return LLMResult(
        text=text,
        model="m",
        tier=ModelTier.DEV,
        provider="fake",
        tool_calls=tool_calls,
        usage=usage,
        cost_est=cost_est,
    )


def _usage(total: int) -> TokenUsage:
    return TokenUsage(input_tokens=total, output_tokens=0, total_tokens=total)


def _accept_soft_check() -> LLMResult:
    return _result(text='{"reject": [], "retry_reconciliation": false}')


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


class _HangTcmComplete:
    """TCM SubAgent hangs; nutrition finishes. SubAgent timeout → unilateral output."""

    def __init__(self) -> None:
        self.tcm_cancelled = False

    async def __call__(self, messages, *, tools=None, **kwargs):
        system_text = (messages[0].get("content") or "") if messages else ""
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if "路由分类器" in system_text:
            return _result(text='{"branch":"full_recommend","domain_hint":null}')
        if "中医饮食 SubAgent" in system_text:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self.tcm_cancelled = True
                raise
        if "营养学 SubAgent" in system_text:
            if has_tool_result:
                return _result(text="营养学侧结论 [source: n1]")
            return _result(
                tool_calls=[ToolCall(id="c1", name="retrieve_nutrition", arguments={"query": "x"})]
            )
        return _accept_soft_check()


def test_subagent_timeout_falls_back_to_single_side_and_cancels_slow_side(monkeypatch):
    monkeypatch.setenv("SUBAGENT_TIMEOUT_S", "0.3")
    complete = _HangTcmComplete()
    server = _server_with_handlers(
        retrieve_nutrition=lambda **kw: [
            {
                "source_id": "n1",
                "domain": "nutrition",
                "source_file": "b",
                "source_type": "t",
                "text": "x",
                "metadata": {},
                "score": 0.5,
            }
        ],
    )
    client = _client_with(server, complete)
    resp = client.post("/api/chat", json={"session_id": "s1", "message": "今天该吃什么"})
    events = _parse_sse(resp.text)
    assert any(e == "guardrail" and "partial_failure" in d for e, d in events)
    token_texts = "".join(d for e, d in events if e == "token")
    assert "营养学侧结论" in token_texts
    assert complete.tcm_cancelled is True


class _BothHangComplete:
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


def test_chain_timeout_cancels_both_subagents_and_emits_sse(monkeypatch):
    monkeypatch.setenv("CHAIN_TIMEOUT_S", "0.4")
    monkeypatch.setenv("SUBAGENT_TIMEOUT_S", "5")
    complete = _BothHangComplete()
    client = _client_with(_server_with_handlers(), complete)
    resp = client.post("/api/chat", json={"session_id": "s1", "message": "今天该吃什么"})
    events = _parse_sse(resp.text)
    assert any(e == "guardrail" and "chain_timeout" in d for e, d in events)
    assert events[-1][0] == "done"
    assert complete.tcm_cancelled is True
    assert complete.nutrition_cancelled is True


class _PricedDualComplete:
    """Each complete() reports usage so the chat span can sum both sides."""

    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(self, messages, *, tools=None, **kwargs):
        self.call_count += 1
        priced = dict(usage=_usage(10), cost_est=0.01)
        system_text = (messages[0].get("content") or "") if messages else ""
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        if "路由分类器" in system_text:
            return _result(text='{"branch":"full_recommend","domain_hint":null}', **priced)
        if "中医饮食 SubAgent" in system_text:
            if has_tool_result:
                return _result(text="阳虚质注意保暖 [source: t1]", **priced)
            return _result(
                tool_calls=[ToolCall(id="c1", name="retrieve_tcm", arguments={"query": "阳虚"})],
                **priced,
            )
        if "营养学 SubAgent" in system_text:
            if has_tool_result:
                return _result(text="适量补充蛋白质 [source: n1]", **priced)
            return _result(
                tool_calls=[
                    ToolCall(id="c2", name="retrieve_nutrition", arguments={"query": "蛋白质"})
                ],
                **priced,
            )
        if "调和层" in system_text:
            return _result(
                text="综合建议：注意保暖同时适量补充蛋白质 [source: t1] [source: n1]",
                **priced,
            )
        return _result(text='{"reject": [], "retry_reconciliation": false}', **priced)


def test_dual_dispatch_request_cost_is_sum_of_both_sides():
    backend = use_memory_backend()
    complete = _PricedDualComplete()
    server = _server_with_handlers(
        retrieve_tcm=lambda **kw: [
            {
                "source_id": "t1",
                "domain": "tcm",
                "source_file": "a",
                "source_type": "t",
                "text": "阳虚忌生冷",
                "metadata": {},
                "score": 0.8,
            }
        ],
        retrieve_nutrition=lambda **kw: [
            {
                "source_id": "n1",
                "domain": "nutrition",
                "source_file": "b",
                "source_type": "t",
                "text": "高蛋白食物",
                "metadata": {},
                "score": 0.7,
            }
        ],
    )
    client = _client_with(server, complete)
    resp = client.post("/api/chat", json={"session_id": "s1", "message": "今天该吃什么"})
    assert resp.status_code == 200
    chat = next(s for s in backend.spans if s.name == "chat")
    assert chat.output["llm_calls"] == complete.call_count
    assert chat.output["tokens"] == complete.call_count * 10
    assert chat.output["cost_est"] == pytest.approx(complete.call_count * 0.01)
    assert chat.output["cost_incomplete"] is False
    assert complete.call_count >= 6


class _HangSingleDomainComplete:
    def __init__(self) -> None:
        self.cancelled = False

    async def __call__(self, messages, *, tools=None, **kwargs):
        system_text = (messages[0].get("content") or "") if messages else ""
        if "路由分类器" in system_text:
            return _result(text='{"branch":"fact_query","domain_hint":"tcm"}')
        if "中医饮食 SubAgent" in system_text:
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return _accept_soft_check()


def test_single_domain_subagent_timeout_emits_guardrail(monkeypatch):
    monkeypatch.setenv("SUBAGENT_TIMEOUT_S", "0.3")
    complete = _HangSingleDomainComplete()
    client = _client_with(_server_with_handlers(), complete)
    resp = client.post("/api/chat", json={"session_id": "s1", "message": "红枣是什么性味"})
    events = _parse_sse(resp.text)
    assert any(e == "guardrail" and "subagent_timeout" in d for e, d in events)
    assert "token" not in [e for e, _ in events]
    assert events[-1][0] == "done"
    assert complete.cancelled is True

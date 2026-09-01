"""Evidence failures use one no-tool repair without rerunning retrieval."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from api.schemas import ChatRequest
from backend.agents._subagent_common import SubAgentResult
from backend.agents.clarification import InMemoryClarificationStore
from backend.agents.dispatch import _stream_dual_dispatch
from backend.agents.reconciliation import ReconciliationResult
from backend.agents.routing import RouteBranch, RouteDecision
from backend.agents.verification import RejectedItem, SuggestionItem, VerificationResult
from backend.llm.adapter import ModelTier
from backend.mcp_server.server import DietExpertMcpServer


def _ok_sub(domain: str, text: str) -> SubAgentResult:
    return SubAgentResult(
        domain=domain,
        final_text=text,
        tool_call_count=1,
        iterations=1,
        terminated_reason="no_tool_use",
        messages=[],
        tools_called=["retrieve_tcm"],
    )


def test_dual_dispatch_repairs_insufficient_evidence_without_rerunning_subagents(monkeypatch):
    calls = {"tcm": 0, "nutrition": 0, "verify": 0, "repair": 0}

    async def fake_tcm(*_a, **_k):
        calls["tcm"] += 1
        return _ok_sub("tcm", f"tcm pass {calls['tcm']} [source: t1]")

    async def fake_nutrition(*_a, **_k):
        calls["nutrition"] += 1
        return _ok_sub("nutrition", f"nutrition pass {calls['nutrition']} [source: n1]")

    async def fake_reconcile(**_k):
        return ReconciliationResult(
            text="调和结论 [source: t1]",
            model="m",
            tier=ModelTier.DEV,
            provider="fake",
        )

    async def fake_verify(*_a, **_k):
        calls["verify"] += 1
        return VerificationResult(
            accepted=[],
            rejected=[
                RejectedItem(
                    item=SuggestionItem(text="草稿"),
                    check_number=1,
                    reason="missing source",
                )
            ],
            needs_reconciliation_retry=True,
        )

    async def fake_repair(*_a, **_k):
        calls["repair"] += 1
        return SuggestionItem(text="保留的回答 [source: t1]")

    monkeypatch.setattr("backend.agents.dispatch.run_tcm_subagent", fake_tcm)
    monkeypatch.setattr("backend.agents.dispatch.run_nutrition_subagent", fake_nutrition)
    monkeypatch.setattr("backend.agents.dispatch.reconcile_subagent_results", fake_reconcile)
    monkeypatch.setattr("backend.agents.dispatch.verify", fake_verify)
    monkeypatch.setattr("backend.agents.dispatch.repair_insufficient_evidence", fake_repair)
    monkeypatch.setattr("backend.agents.dispatch.check_allergens", lambda *_a, **_k: [])
    monkeypatch.setattr("backend.agents.dispatch.record_conflict_gap", lambda **_k: True)

    async def collect():
        events = []
        async for chunk in _stream_dual_dispatch(
            ChatRequest(session_id="s1", message="今天吃什么"),
            RouteDecision(branch=RouteBranch.FULL_RECOMMEND, reason="test"),
            DietExpertMcpServer(tools={}),
            complete=AsyncMock(),
            trace_id="tr-retry",
            profile=None,
            conflict_rules_fetcher=lambda *_a: [],
            clarification_store=InMemoryClarificationStore(),
        ):
            events.append(chunk)
        return events

    events = asyncio.run(collect())
    assert calls == {"tcm": 1, "nutrition": 1, "verify": 1, "repair": 1}
    assert any("token" in e for e in events)


def test_dual_dispatch_repairs_citation_relevance_without_second_retrieval(monkeypatch):
    """Unsupported citation relevance is repaired in-place, without a second pass."""
    calls = {"tcm": 0, "nutrition": 0, "verify": 0, "repair": 0}

    async def fake_tcm(*_a, **_k):
        calls["tcm"] += 1
        return _ok_sub("tcm", f"tcm pass {calls['tcm']} [source: t1]")

    async def fake_nutrition(*_a, **_k):
        calls["nutrition"] += 1
        return _ok_sub("nutrition", f"nutrition pass {calls['nutrition']} [source: n1]")

    async def fake_reconcile(**_k):
        return ReconciliationResult(
            text="调和结论 [source: t1]",
            model="m",
            tier=ModelTier.DEV,
            provider="fake",
        )

    async def fake_verify(*_a, **_k):
        calls["verify"] += 1
        return VerificationResult(
            accepted=[],
            rejected=[
                RejectedItem(
                    item=SuggestionItem(text="草稿"),
                    check_number=2,
                    reason="引用的 source_id 仅支持某个细节，不支持整体结论",
                )
            ],
        )

    async def fake_repair(*_a, **_k):
        calls["repair"] += 1
        return SuggestionItem(text="保留的部分 [source: t1]")

    monkeypatch.setattr("backend.agents.dispatch.run_tcm_subagent", fake_tcm)
    monkeypatch.setattr("backend.agents.dispatch.run_nutrition_subagent", fake_nutrition)
    monkeypatch.setattr("backend.agents.dispatch.reconcile_subagent_results", fake_reconcile)
    monkeypatch.setattr("backend.agents.dispatch.verify", fake_verify)
    monkeypatch.setattr("backend.agents.dispatch.repair_insufficient_evidence", fake_repair)
    monkeypatch.setattr("backend.agents.dispatch.check_allergens", lambda *_a, **_k: [])
    monkeypatch.setattr("backend.agents.dispatch.record_conflict_gap", lambda **_k: True)

    async def collect():
        events = []
        async for chunk in _stream_dual_dispatch(
            ChatRequest(session_id="s1", message="今天吃什么"),
            RouteDecision(branch=RouteBranch.FULL_RECOMMEND, reason="test"),
            DietExpertMcpServer(tools={}),
            complete=AsyncMock(),
            trace_id="tr-retry-check2",
            profile=None,
            conflict_rules_fetcher=lambda *_a: [],
            clarification_store=InMemoryClarificationStore(),
        ):
            events.append(chunk)
        return events

    events = asyncio.run(collect())
    assert calls == {"tcm": 1, "nutrition": 1, "verify": 1, "repair": 1}
    assert any("token" in e for e in events)


def test_forced_clarification_retry_degrades_to_successful_side(monkeypatch):
    """A side that keeps asking for clarification must not discard the other side."""
    calls = {"tcm": 0, "nutrition": 0, "verify": 0, "reconcile": 0}

    async def fake_tcm(*_a, **_k):
        calls["tcm"] += 1
        if calls["tcm"] == 1:
            return _ok_sub("tcm", "[NEED_CLARIFICATION] 请说明具体食物")
        return _ok_sub("tcm", "TCM forced answer [source: t1]")

    async def fake_nutrition(*_a, **_k):
        calls["nutrition"] += 1
        return _ok_sub("nutrition", "[NEED_CLARIFICATION] Please name the food")

    async def fake_reconcile(**_k):
        calls["reconcile"] += 1
        raise AssertionError("single-sided fallback must not call reconciliation")

    async def fake_verify(*_a, **_k):
        calls["verify"] += 1
        return VerificationResult(
            accepted=[SuggestionItem(text="TCM forced answer [source: t1]")],
            rejected=[],
        )

    monkeypatch.setattr("backend.agents.dispatch.run_tcm_subagent", fake_tcm)
    monkeypatch.setattr("backend.agents.dispatch.run_nutrition_subagent", fake_nutrition)
    monkeypatch.setattr("backend.agents.dispatch.reconcile_subagent_results", fake_reconcile)
    monkeypatch.setattr("backend.agents.dispatch.verify", fake_verify)
    monkeypatch.setattr("backend.agents.dispatch.check_allergens", lambda *_a, **_k: [])

    async def collect():
        events = []
        async for chunk in _stream_dual_dispatch(
            ChatRequest(session_id="s1", message="这个能不能吃"),
            RouteDecision(branch=RouteBranch.CANDIDATE_EVAL, reason="test"),
            DietExpertMcpServer(tools={}),
            complete=AsyncMock(),
            trace_id="tr-partial-clarification",
            profile=None,
            conflict_rules_fetcher=lambda *_a: [],
            clarification_store=InMemoryClarificationStore(),
            allow_clarification=False,
        ):
            events.append(chunk)
        return events

    events = asyncio.run(collect())
    token_text = "".join(data for event in events if (data := event) and "event: token" in event)
    assert calls == {"tcm": 2, "nutrition": 2, "verify": 1, "reconcile": 0}
    assert any("partial_failure" in event and "仅展示中医侧结论" in event for event in events)
    assert "TCM forced answer" in token_text
    assert not any("clarification_unresolved" in event for event in events)

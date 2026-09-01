"""
测试目标：api/main.py `_parse_sse_chunk()`/`_TurnAccumulator` —— D27 补充
(backend/memory/session_store.py 接线)里，把 dispatch_branch()/stream_multi_task()
吐出的原始 SSE 事件流重建成 backend/memory/compression.py `TurnRecord` 的逻辑。
对应实现：api/main.py
覆盖要求：纯逻辑，不碰数据库/网络。
"""
from __future__ import annotations

from api.main import _TurnAccumulator, _parse_sse_chunk
from backend.agents.sse import sse_event


def test_parse_sse_chunk_extracts_event_and_data():
    chunk = sse_event("token", {"text": "hello"})
    event, data = _parse_sse_chunk(chunk)
    assert event == "token"
    assert data == {"text": "hello"}


def test_parse_sse_chunk_handles_unicode_content():
    chunk = sse_event("token", {"text": "红枣性温"})
    event, data = _parse_sse_chunk(chunk)
    assert data["text"] == "红枣性温"


def test_parse_sse_chunk_returns_none_for_malformed_input():
    event, data = _parse_sse_chunk("not an sse chunk at all")
    assert event is None
    assert data == {}


def test_accumulator_builds_conclusion_from_token_chunks():
    acc = _TurnAccumulator()
    acc.observe(sse_event("token", {"text": "红枣"}))
    acc.observe(sse_event("token", {"text": "性温"}))
    acc.observe(sse_event("done", {"trace_id": "x"}))
    turn = acc.build(branch_fallback="fact_query", user_text="红枣是什么性味")
    assert turn.conclusion == "红枣性温"
    assert turn.branch == "fact_query"
    assert "红枣是什么性味" in turn.raw_text
    assert "红枣性温" in turn.raw_text


def test_accumulator_collects_cited_source_ids_deduplicated():
    acc = _TurnAccumulator()
    acc.observe(sse_event("source", {"source_id": "t1"}))
    acc.observe(sse_event("source", {"source_id": "t1"}))  # 重复引用只算一次
    acc.observe(sse_event("source", {"source_id": "t2"}))
    turn = acc.build(branch_fallback="fact_query", user_text="x")
    assert turn.cited_source_ids == ("t1", "t2")


def test_accumulator_collects_guardrail_types():
    acc = _TurnAccumulator()
    acc.observe(sse_event("guardrail", {"type": "partial_failure", "detail": "..."}))
    turn = acc.build(branch_fallback="full_recommend", user_text="x")
    assert turn.triggered_guardrails == ("partial_failure",)


def test_accumulator_collects_rejected_item_reasons():
    acc = _TurnAccumulator()
    acc.observe(sse_event("guardrail", {"type": "verification_rejected", "detail": "..."}))
    acc.observe(sse_event("guardrail", {"type": "rejected_item", "check_number": 4, "reason": "命中过敏原"}))
    turn = acc.build(branch_fallback="full_recommend", user_text="x")
    assert turn.rejected_suggestions == ("命中过敏原",)
    assert "verification_rejected" in turn.triggered_guardrails
    assert "rejected_item" in turn.triggered_guardrails


def test_accumulator_uses_branch_fallback_when_no_task_events_seen():
    """单任务路径不会有 task 事件，branch 必须靠调用方传入的 fallback。"""
    acc = _TurnAccumulator()
    acc.observe(sse_event("token", {"text": "x"}))
    turn = acc.build(branch_fallback="single_domain", user_text="x")
    assert turn.branch == "single_domain"


def test_accumulator_merges_multiple_task_branches_for_multi_task_turn():
    """D32 多任务场景：多个 task 事件的 branch 拼接成一个，不按子任务拆分成
    多条记录(见 session_store.py 模块文档"已知限制")。"""
    acc = _TurnAccumulator()
    acc.observe(sse_event("task", {"index": 0, "total": 2, "branch": "log_write", "text": "..."}))
    acc.observe(sse_event("token", {"text": "已记录"}))
    acc.observe(sse_event("task_done", {"index": 0}))
    acc.observe(sse_event("task", {"index": 1, "total": 2, "branch": "single_domain", "text": "..."}))
    acc.observe(sse_event("token", {"text": "阳虚质建议"}))
    acc.observe(sse_event("task_done", {"index": 1}))
    turn = acc.build(branch_fallback="ignored", user_text="x")
    assert turn.branch == "log_write+single_domain"
    assert turn.conclusion == "已记录阳虚质建议"
